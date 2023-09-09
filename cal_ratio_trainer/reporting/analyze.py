from pathlib import Path
from typing import Dict, List

from attr import dataclass
from cal_ratio_trainer.config import AnalyzeConfig, training_spec
from cal_ratio_trainer.utils import HistoryTracker, find_training_result


@dataclass
class TrainingEpoch:
    """A single training epoch"""

    # Name of the run
    run_name: str

    # Path to the running directory
    run_dir: Path

    # Epoch number
    epoch: int

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
            run_name=run_dir.name,
            run_dir=run_dir,
            epoch=e,
            history=epoch_h.values_for(e),
        )
        for e_list in [
            best_epochs_nn_loss,
            best_epochs_adv_loss,
            best_epochs_disc_loss,
            best_epochs_ks,
        ]
        for e in e_list
    ]


def analyze_training_runs(cache: Path, config: AnalyzeConfig):
    """Get and dump the most interesting runs from the list of runs we've done.

    Args:
        cache (Path): Where cache data files are located
        config (AnalyzeConfig): Config to steer our work
    """
    assert config.runs_to_analyze is not None
    best_results = [get_best_results(c) for c in config.runs_to_analyze]

    print(best_results)
