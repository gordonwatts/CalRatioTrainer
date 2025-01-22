from pathlib import Path
import numpy as np

import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

from cal_ratio_trainer.common.trained_model import TrainedModel, load_model_from_spec
from cal_ratio_trainer.config import ScorePickleConfig
from cal_ratio_trainer.reporting.evaluation_utils import load_test_data_from_df


def _score_pickle_file(model: TrainedModel, file_path: Path):
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
    ak.to_parquet(ak_data, file_path.parent / f"{file_path.stem}-score.parquet")

    print(predict[:, 0])
    print(predict[:, 1])
    print(predict[:, 2])
    print(f"label 0: { np.sum(predict[:, 0] > 0.1 ) }")
    print(f"label 1: { np.sum(predict[:, 1] > 0.1 ) }")
    print(f"label 2: { np.sum(predict[:, 2] > 0.1 ) }")
    print(f"length: {len(predict[:, 0])}")

    tot_correct = 0
    for x in predict:
        if (x[1] > x[0]) & (x[1] > x[2]):
            tot_correct += 1
    score = tot_correct / len(predict)
    print("NN Performance Score - Frac Max Sig ", score)

    tot_correct = 0
    for x in predict:
        if x[1] > 0.5:
            tot_correct += 1
    score = tot_correct / len(predict)
    print("NN Performance Score - Frac Sig % > 50 ", score)

    tot_correct = 0
    for x in predict:
        if x[1] > 0.8:
            tot_correct += 1
    score = tot_correct / len(predict)
    print("NN Performance Score - Frac Sig % > 80 ", score)

    plot_score(
        data_dict.get("jet_nn_qcd"),
        data_dict.get("jet_nn_bib"),
        data_dict.get("jet_nn_sig"),
        file_path,
    )


def score_pkl_files(config: ScorePickleConfig):
    """Score a list of pickle files, writing out
    parquet files with the scored jets.

    Args:
        config (ScorePickleConfig): Config describing parameters for the job.
    """
    # Load the model for the training we have been given.
    assert config.training is not None

    # Load the training
    model = load_model_from_spec(config.training)

    # Run through each file
    assert config.input_files is not None
    for f in config.input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file {f} does not exist")

        _score_pickle_file(model, f)


def plot_score(qcd_pred, bib_pred, sig_pred, file_path):
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
    # ax.set_title("title")
    # plt.yscale("log")
    ax.set_xlim(0.00002, 1.0)
    ax.set_ylim(top=1.2)
    # ax.set_ylim(top=50)
    ax.legend(loc="upper right")
    #
    plt.savefig(file_path.parent / f"{file_path.stem}-nn_score_plot.png")
    plt.clf()
    plt.close(fig)

    return
