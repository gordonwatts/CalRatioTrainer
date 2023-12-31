from pathlib import Path
from typing import Any, Dict, List, Tuple

from attr import dataclass
from matplotlib import pyplot as plt
import yaml
from cal_ratio_trainer.common.column_names import EventType
from cal_ratio_trainer.common.evaluation import (
    normalize_to_one,
    plot_roc_curve,
    signal_llp_efficiencies,
)
from cal_ratio_trainer.common.trained_model import load_trained_model_from_training

from cal_ratio_trainer.config import AnalyzeConfig, training_spec
from cal_ratio_trainer.reporting.evaluation_utils import (
    TrainedModelData,
    load_test_data,
)
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

    @property
    def nickname(self) -> str:
        """Get a nickname for this epoch"""
        return f"{self.run_name}/{self.epoch}"


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


def _get_epoch_plot(e_info: TrainingEpoch, plot_name_stub: str) -> Path:
    """Get the path for a specific plot from the training

    Args:
        e_info (TrainingEpoch): The training epoch to fetch
        plot_name_stub (str): The plot name stub to use

    Returns:
        Path: Location of the plot
    """
    # Get the path to the plot
    plot_path = e_info.run_dir / f"{e_info.epoch:03d}_{plot_name_stub}.png"
    if not plot_path.exists():
        raise ValueError(f"Plot path {plot_path} does not exist.")

    return plot_path


def _analyze_auc(
    data: TrainedModelData, path: Path, epoch: int, report: MDReport
) -> str:
    "Return the calculated AUC for this epoch and a link to the plot"

    model = load_trained_model_from_training(path, epoch)
    predictions = model.predict(data)

    # Balance the weights
    normalized_weights = normalize_to_one(data.weights[0], data.y.values[0])

    auc, fig = plot_roc_curve(
        None,
        normalized_weights,
        predictions,
        third_label=EventType.BIB.value,
        threshold=0,
        y_test=data.y.values,  # type: ignore
        label_string="my foot",
    )

    plt.title(f"AUC Curve for {path.parent.name}/{path.name}/{epoch} (test data)")

    r = report.figure_link(fig, f"{auc:.3f}")
    plt.close(fig)
    return r


class HeldData:
    """Holds the data for various runs"""

    def __init__(self):
        self._data: Dict[str, TrainedModelData] = {}

    def __str__(self) -> str:
        return f"HeldData(len={len(self._data)})"

    def __getitem__(self, key: Path) -> TrainedModelData:
        if str(key) not in self._data:
            self._data[str(key)] = load_test_data(key)
        return self._data[str(key)]


def analyze_training_runs(cache: Path, config: AnalyzeConfig):
    """Get and dump the most interesting runs from the list of runs we've done.

    Args:
        cache (Path): Where cache data files are located
        config (AnalyzeConfig): Config to steer our work
    """
    # Grab a list of "interesting" epochs and training and then sort them
    # by the main loss so the eye can read them easily.
    assert config.runs_to_analyze is not None
    best_results_lists = [get_best_results(c) for c in config.runs_to_analyze]
    best_results_collection = [r for r_list in best_results_lists for r in r_list]
    best_results = sorted(
        best_results_collection, key=lambda r: r.history["val_original_lossf"]
    )

    # Create the report file
    assert config.output_report is not None
    with MDReport(config.output_report, "Training Analysis") as report:
        report.write(
            "Summary of the better epochs (validation sample), sorted by main "
            "network loss on validation sample."
        )
        # Want to print out the best_results
        held_data = HeldData()
        t_list = [
            {
                "Name": r.nickname,
                "Reason": r.reason,
                "main loss": f"{r.history['val_original_lossf']:.4f}",
                "K-S Sum": f"{r.history['ks']:.4f}",
                "Adversary Loss": f"{r.history['val_original_adv_lossf']:.4f}",
                "Discriminator Loss": f"{r.history['val_adv_loss']:.4f}",
                "AUC": _analyze_auc(held_data[r.run_dir], r.run_dir, r.epoch, report),
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
        report.header("## Loss and Accuracy Values For All Epochs")
        summary_values_table = [
            {
                "Name": k,
                **{r.nickname: f"{r.history[k]:.03f}" for r in best_results},
            }
            for k in sorted(best_results[0].history.keys())
        ]
        report.add_table(summary_values_table)

        # Lets go through and evaluate each model against the test data.
        eff: Dict[str, Dict[Tuple[int, int], float]] = {}
        eff_plots: Dict[str, str] = {}
        for r in best_results:
            # Load up the test data for this epoch
            data = held_data[r.run_dir]

            # Generate predictions for this run
            model = load_trained_model_from_training(r.run_dir, r.epoch)
            predictions = model.predict(data)

            # Get the info back (along with a figure):
            fig, pass_frac = signal_llp_efficiencies(predictions, data.y, data.z)
            eff_plots[r.nickname] = report.figure_link(fig, "Signal Efficiency")
            plt.close(fig)
            eff[r.nickname] = pass_frac

        # Now, lets make a table of the efficiencies. There is a column for each item
        # in eff, and a row for each mH, mS combination.
        report.header("## Signal Efficiencies")
        list_of_mass_combos = list(eff.values())[0].keys()
        eff_table = [
            {
                "Mass": f"{int(mH)}, {int(mS)}",
                **{k: f"{eff[k][(mH, mS)]:.03f}" for k in sorted(eff.keys())},
            }
            for mH, mS in list_of_mass_combos
        ]
        eff_table.append(
            {
                "Mass": "",
                **{k: eff_plots[k] for k in sorted(eff.keys())},
            }
        )
        report.add_table(eff_table, bold_max_col_value=True)

        # Now some plots from each of the best runs.
        report.header("## Plots From Each Run")
        summary_plots_table = [
            {
                "Name": k[1],
                **{
                    r.nickname: report.figure_md(_get_epoch_plot(r, k[0]))
                    for r in best_results
                },
            }
            for k in [
                ("main_sig_predictions_linear", "Main Signal Prediction"),
                ("main_qcd_predictions_linear", "Main QCD Prediction"),
                ("main_bib_predictions_linear", "Main BIB Prediction"),
                ("val_adversary_sig_predictions", "Adversary Sig Prediction"),
                ("val_adversary_qcd_predictions", "Adversary QCD Prediction"),
                ("val_adversary_bib_predictions", "Adversary BIB Prediction"),
            ]
        ]
        report.add_table(summary_plots_table)

        # Next, list the parameters for each run in two tables. First table are all the
        # parameters that are different. And the second is just a listing of the
        # remaining parameters that are the same.
        report.header("## Training Parameters")

        # First, get a list of all the parameters that are different.
        def load_params(run_dir: Path) -> Dict[str, Any]:
            with (run_dir / "training_params.yaml").open("r") as f:
                return yaml.safe_load(f)

        training_params = [(rd, load_params(rd)) for rd in all_runs]

        # Which ones are not the same across it all and which ones are different?
        param_is_same = {
            k: len(set([str(p[k]) for _, p in training_params])) == 1
            for k in training_params[0][1].keys()
        }

        # Now, dump those that are different in a table.
        if not all(param_is_same.values()):
            report.write("Parameters That Vary across runs")
            different_params = [
                {
                    **{"Run": str(rd)},
                    **{k: str(p[k]) for k in p.keys() if not param_is_same[str(k)]},
                }
                for rd, p in training_params
            ]
            report.add_table(different_params)

        # And single column dump of those parameters that are the same.
        report.write("Parameters common for all runs:")
        same_params = [
            {
                "Name:": k,
                "Value": str(training_params[0][1][k]),
            }
            for k in training_params[0][1].keys()
            if param_is_same[k]
        ]
        report.add_table(same_params)
