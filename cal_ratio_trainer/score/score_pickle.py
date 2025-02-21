from pathlib import Path
import numpy as np

import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

from cal_ratio_trainer.common.trained_model import TrainedModel, load_model_from_spec
from cal_ratio_trainer.config import ScorePickleConfig
from cal_ratio_trainer.reporting.evaluation_utils import load_test_data_from_df
from cal_ratio_trainer.reporting.md_report import MDReport


def _score_pickle_file(model: TrainedModel, file_path: Path, output_path: Path):
    """Run the ML scoring on the data in file_path.

    Args:
        file_path (Path): The path to the input pickle file.
    """
    # Load the data and format it for training. Note
    # that we assume signal - the label is never used.
    data = pd.read_pickle(file_path)
    assert isinstance(data, pd.DataFrame)
    training_data = load_test_data_from_df(data)

    # Run the prediction
    predict = model.predict(training_data)

    # Build the output and write it to parquet. For comparison reasons,
    # we need run, event, jet pt, eta, and phi, along with the three
    # prediction outputs. We'll create an awkward event.
    data_dict = {
        # "run_number": data["run_number"],
        # "event_number": data["event_number"],
        "jet_pt": data["jet_pt"],
        "jet_eta": data["jet_eta"],
        "jet_phi": data["jet_phi"],
        "label": data["label"],
        "jet_nn_qcd": predict[:, 0],
        "jet_nn_sig": predict[:, 1],
        "jet_nn_bib": predict[:, 2],
    }

    ak_data = ak.from_iter(data_dict)
    ak.to_parquet(ak_data, f"{output_path.parts[0]}/score.parquet")

    return data_dict


def score_pkl_files(config: ScorePickleConfig):
    """Score a list of pickle files, writing out
    parquet files with the scored jets.

    Args:
        config (ScorePickleConfig): Config describing parameters for the job.
    """
    # Load the model for the training we have been given.
    assert config.training is not None

    # Create directory where we're putting the markdown file and saving the scores
    assert config.output_path is not None
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the training
    model = load_model_from_spec(config.training)

    # Run through each file
    assert config.input_files is not None
    for f in config.input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file {f} does not exist")

        data_dict = _score_pickle_file(model, f, config.output_path)
        make_md_score_report(config, data_dict)


def plot_score(qcd_pred, bib_pred, sig_pred):
    """
    Outputs a plot of the 3 different NN scores for the file

    Args:
        qcd_pred/bib_pred/sig_pred: model prediction for sig/qcd/bib
    """

    bin_list = np.linspace(0, 1, 30)

    n_qcd, bin_edges_qcd = np.histogram(qcd_pred, bins=bin_list)
    n_bib, bin_edges_bib = np.histogram(bib_pred, bins=bin_list)
    n_sig, bin_edges_sig = np.histogram(sig_pred, bins=bin_list)

    n_qcd = n_qcd / np.sum(n_qcd)
    n_bib = n_bib / np.sum(n_bib)
    n_sig = n_sig / np.sum(n_sig)

    bin_centers_sig = (bin_edges_sig[:-1] + bin_edges_sig[1:]) / 2.0

    fig, ax = plt.subplots()

    ax.bar(
        bin_centers_sig,
        n_sig,
        width=np.diff(bin_list),
        color="blue",
        alpha=0.5,
        label="SIGNAL NN SCORE",
        align="center",
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="QCD NN SCORE")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB NN SCORE")

    ax.set_xlabel("NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")
    # plt.yscale("log")
    ax.set_xlim(0.00002, 1.0)
    ax.set_ylim(top=1.2)
    # ax.set_ylim(top=50)
    ax.legend(loc="upper right")

    return fig


def make_md_score_report(config: ScorePickleConfig, data_dict: dict):
    """
    Makes a markdown report of the result of the score function
    Includes information about which model was used, which test dataset was used,
    A plot showing the NN score distribution
    and some information about the % of signal NN scores above certain cuts
    """
    assert config.output_path is not None
    with MDReport(config.output_path, "Score Report") as report:
        report.header("## Model and Test Dataset Info")
        report.write(
            "Path to training dataset and model information used in making this scoring report"
        )
        report.add_table(
            [
                {
                    "path to test dataset": str(config.input_files[0]),
                    "model": config.training,
                }
            ]
        )

        report.header("## Score Results")

        report.write("Information resulting from the scoring function")

        num_max_sig = 0
        num_greater_fifty = 0
        num_greater_eighty = 0

        tot_length = len(data_dict.get("jet_nn_qcd"))

        for x in range(tot_length):
            if (data_dict.get("jet_nn_sig")[x] > data_dict.get("jet_nn_bib")[x]) & (
                data_dict.get("jet_nn_sig")[x] > data_dict.get("jet_nn_qcd")[x]
            ):
                num_max_sig += 1
            if data_dict.get("jet_nn_sig")[x] > 0.5:
                num_greater_fifty += 1
            if data_dict.get("jet_nn_sig")[x] > 0.8:
                num_greater_eighty += 1

        report.write("")
        report.write(
            "NN Performance Score - Frac Max Sig: "
            + str(round(num_max_sig / tot_length, 4))
        )
        report.write("")
        report.write(
            "NN Performance Score - Frac Sig % > 50: "
            + str(round(num_greater_fifty / tot_length, 4))
        )
        report.write("")
        report.write(
            "NN Performance Score - Frac Sig % > 80: "
            + str(round(num_greater_eighty / tot_length, 4)),
        )

        report.header("### Score Plot")
        p = plot_score(
            data_dict.get("jet_nn_qcd"),
            data_dict.get("jet_nn_bib"),
            data_dict.get("jet_nn_sig"),
        )
        report.add_figure(p, 600)
        plt.savefig(config.output_path.parent / "score_plot.png")
        plt.close(p)
