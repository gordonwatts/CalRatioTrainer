from pathlib import Path
from typing import Dict, List

from attr import dataclass

from cal_ratio_trainer.config import AnalyzeConfig, training_spec
from cal_ratio_trainer.utils import HistoryTracker, find_training_result

from .md_report import MDReport


@dataclass
class TrainingEpoch:
    """A single training epoch"""

    # Name of the run
    run_name: str

    # Path to the running directory
    run_dir: Path

    # Epoch number
    epoch: int

    # Reason we picked this guy
    reason: str

    # All history numbers
    history: Dict[str, float]


def get_best_results(config: training_spec) -> List[TrainingEpoch]:
    # Find the directory for this training result.
    run_dir = find_training_result(config.name, config.run)
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist.")

    # Now load the history tracker
    epoch_h = HistoryTracker(run_dir / "keras" / "history_checkpoint")

    # Find the best epochs for a few things. First, validation loss for main training.
    num = 2
    best_epochs_nn_loss = epoch_h.find_smallest("val_original_lossf", num)

    # Best loss for the adversary
    best_epochs_adv_loss = epoch_h.find_smallest("val_original_adv_lossf", num)

    # Best loss for the discriminator
    best_epochs_disc_loss = epoch_h.find_smallest("val_adv_loss", num)

    # And for the sum of the ks_test results.
    epoch_h.make_sum("ks", ["ks_qcd_hist", "ks_bib_hist", "ks_sig_hist"])
    best_epochs_ks = epoch_h.find_smallest("ks", num)

    # Finally, we can create the resulting listing.
    return [
        TrainingEpoch(
            run_name=f"{run_dir.parent.name}/{run_dir.name}",
            run_dir=run_dir,
            epoch=e,
            history=epoch_h.values_for(e),
            reason=e_list[1],
        )
        for e_list in [
            (best_epochs_nn_loss, "Best NN Loss"),
            (best_epochs_adv_loss, "Best Adversary Loss"),
            (best_epochs_disc_loss, "Best Discriminator Loss"),
            (best_epochs_ks, "Best K-S Sum"),
        ]
        for e in e_list[0]
    ]


def analyze_training_runs(cache: Path, config: AnalyzeConfig):
    """Get and dump the most interesting runs from the list of runs we've done.

    Args:
        cache (Path): Where cache data files are located
        config (AnalyzeConfig): Config to steer our work
    """
    assert config.runs_to_analyze is not None
    best_results_lists = [get_best_results(c) for c in config.runs_to_analyze]
    best_results = [r for r_list in best_results_lists for r in r_list]

    # Create the report file
    assert config.output_report is not None
    with MDReport(config.output_report, "Training Analysis") as report:
        report.write("Summary of the better epochs (validation sample)")
        # Want to print out the best_results
        t_list = [
            {
                "Name": f"{r.run_name}, epoch {r.epoch}",
                "Reason": r.reason,
                "main loss": f"{r.history['val_original_lossf']:.4f}",
                "K-S Sum": f"{r.history['ks']:.4f}",
                "Adversary Loss": f"{r.history['val_original_adv_lossf']:.4f}",
                "Discriminator Loss": f"{r.history['val_adv_loss']:.4f}",
            }
            for r in best_results
        ]
        col_headings = list(t_list[0].keys())
        col_headings.remove("Name")
        col_headings.remove("Reason")
        col_headings = ["Name", "Reason", *col_headings]

        report.add_table(t_list, col_order=col_headings)

        # Add in the main nn loss and ks loss figures for all the epochs and runs.
        all_runs = set([r.run_dir for r in best_results])
        plot_table = [
            {
                "Run": f"{r.parent.name}/{r.name}",
                "Main Loss": report.figure_md(r / "main_nn_loss.png"),
                "Discriminator Loss": report.figure_md(r / "test_adv_loss.png"),
            }
            for r in all_runs
        ]
        report.add_table(
            plot_table, col_order=["Run", "Main Loss", "Discriminator Loss"]
        )

        # Now some plots from each of the best runs.
        for r in best_results:
            report.header(f"## Run {r.run_name}, epoch {r.epoch} ({r.reason})")
