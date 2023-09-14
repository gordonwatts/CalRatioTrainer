import logging
from io import TextIOWrapper
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, roc_curve


def signal_llp_efficiencies(
    prediction: np.ndarray,
    y_test: pd.Series,
    Z_test: pd.DataFrame,
    f: Optional[TextIOWrapper] = None,
) -> Tuple[Figure, Dict[Tuple[int, int], float]]:
    """Plot signal efficiency as function of mH, mS

    :param prediction: Outputs of NN
    :param y_test: actual labels
    :param Z_test: mH, mS
    :param destination: where to save file
    :param f: output file for text
    """
    sig_rows = np.where(y_test == 1)
    prediction = prediction[sig_rows]
    Z_test = Z_test.iloc[sig_rows]  # type: ignore
    mass_array = (
        Z_test.groupby(["llp_mH", "llp_mS"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    plot_x = []
    plot_y = []
    plot_z = []

    # Loop over all mH, mS combinations
    for item, mH, mS in zip(
        mass_array["count"], mass_array["llp_mH"], mass_array["llp_mS"]
    ):
        temp_array = prediction[(Z_test["llp_mH"] == mH) & (Z_test["llp_mS"] == mS)]
        temp_max = np.argmax(temp_array, axis=1)
        temp_num_signal_best = len(temp_max[temp_max == 1])
        temp_eff = temp_num_signal_best / temp_array.shape[0]
        plot_x.append(mH)
        plot_y.append(temp_eff)
        plot_z.append(mS)
        logging.info("mH: " + str(mH) + ", mS: " + str(mS) + ", Eff: " + str(temp_eff))
        if f is not None:
            f.write("%s,%s,%s\n" % (str(mH), str(mS), str(temp_eff)))

    # Make a nice 2D plot of all of this
    plt.clf()
    fig = plt.figure()
    plt.scatter(
        plot_x,
        plot_y,
        marker="+",  # type: ignore
        s=150,
        linewidths=4,
        c=plot_z,
        cmap=plt.cm.coolwarm,  # type: ignore
    )
    plt.ylim(0.0, 1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"mS")
    plt.xlabel("mH")
    plt.ylabel("Signal Efficiency")

    # Build a dictionary that can be easily used outside.
    data = {(mH, mS): eff for mH, mS, eff in zip(plot_x, plot_z, plot_y)}

    return fig, data


def plot_roc_curve(
    f: TextIOWrapper,
    mcWeights_test: np.ndarray,
    prediction: np.ndarray,
    third_label: int,
    threshold: int,
    y_test: np.ndarray,
    label_string: str,
    n_folds: Optional[int],
) -> Tuple[float, Figure]:
    """Plots a roc curve

    :param destination: where to save plots
    :param f:
    :param mcWeights_test: weights of jets
    :param prediction: NN outputs
    :param third_label: deprecated
    :param threshold: deprecated
    :param y_test: actual jet labels
    :param label_string: what to put on plots
    :param n_folds: which kfold
    :return:
    """
    test_threshold, leftovers = find_threshold(
        prediction, y_test, mcWeights_test, threshold * 100, third_label
    )
    # Make ROC curve of leftovers, those not tagged by above function
    if threshold == 0:
        leftovers = y_test > -1
    bkg_eff, tag_eff, roc_auc = make_multi_roc_curve(
        prediction,
        y_test,
        mcWeights_test,
        test_threshold,
        third_label,
        leftovers,
        n_folds,
    )
    # TODO: uncomment rest
    # Write AUC to training_details.txt
    f.write(
        "Threshold: %s, ROC AUC: %s, label: %s\n"
        % (str(-threshold + 1), str(roc_auc), str(label_string))
    )
    # Make ROC curve
    fig = plt.figure()
    plt.plot(
        tag_eff,
        bkg_eff,
        label=str(label_string)
        + f", BIB Eff: {threshold :.3f}"
        + f", AUC: {roc_auc:.3f}",
    )
    plt.xlabel("LLP Tagging Efficiency")
    axes = plt.gca()
    axes.set_xlim([0, 1])

    plt.legend()
    plt.yscale("log")
    y_label = "BIB Rejection" if third_label == 2 else "QCD Rejection"
    plt.ylabel(y_label)

    return roc_auc, fig


def find_threshold(prediction, y, weight, perc, label):
    """Function to find threshold weight at which you get percentage (perc) right

    :param prediction: Inputs for NN jet output
    :param y: the truth
    :param weight: weight of input, not currently needed
    :param perc: Percent aimed for
    :param label: Label we are using in ROC curve
    :return:
    """
    # Instead of lame loops let's order our data, then find percentage from there
    # prediction is 3xN, want to sort by BIB weight

    label_events_y = y[y == label]
    label_events_prediction = prediction[y == label]

    prediction_sorted = np.array(
        label_events_prediction[label_events_prediction[:, label].argsort()]
    )

    cutoffIndex = (round(((100 - perc) / 100) * label_events_y.size)) - 1
    threshold = prediction_sorted.item((int(cutoffIndex), label))

    leftovers = np.where(np.greater(threshold, prediction[:, label]))

    return threshold, leftovers


def make_multi_roc_curve(
    prediction, y, weight, threshold, label, leftovers, n_folds
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Make family of ROC Curves
       Since this is a 3-class problem, first make cut on BIB weight for given
       percentage of correctly tagged BIB
       Take all leftover, and make ROC curve with those
       Make sure to take into account signal and QCD lost in BIB tagged jets

    :param prediction: jet NN output
    :param y: jet label
    :param weight: jet weight
    :param threshold: deprecated
    :param label: deprecated
    :param leftovers: jets not in ROC curve
    :param n_folds: which kfold
    :return:
    """
    # Leftover are the indices of jets left after taking out jets below BIB cut
    # So take all those left
    prediction_left = prediction[leftovers]
    y_left = y[leftovers]
    weight_left = weight[leftovers]

    # Find signal_ratio and qcd_ratio and bib_ratio, ratio of how many signal
    # or qcd or bib left after BIB cut vs how many there were originally
    num_signal_original = y[y == 1].size
    num_signal_leftover = y_left[y_left == 1].size
    signal_ratio = num_signal_leftover / num_signal_original

    num_qcd_original = np.sum(weight[y == 0])
    num_qcd_leftover = np.sum(weight_left[y_left == 0])
    qcd_ratio = num_qcd_leftover / num_qcd_original

    num_bib_original = np.sum(weight[y == 2])
    num_bib_leftover = np.sum(weight_left[y_left == 2])
    bib_ratio = num_bib_leftover / num_bib_original

    # BIB weight 0.1
    weight_left[y_left == 2] = weight_left[y_left == 2] * 0.1

    prediction_left_signal = prediction_left[:, 1]

    # If we are looking at BIB cut, then signal vs QCD roc curve
    # Use roc_curve function from scikit-learn
    if label == 2:
        (fpr, tpr, _) = roc_curve(
            y_left, prediction_left_signal, sample_weight=weight_left, pos_label=1
        )
        # Scale results by qcd_ratio, signal_ratio
        a = auc(fpr * qcd_ratio, tpr * signal_ratio)

        # return results of roc curve
        with np.errstate(divide="ignore"):
            goodIndices = np.where(np.isfinite(1 / fpr))
        return (
            (1 / fpr[goodIndices]) * qcd_ratio,
            tpr[goodIndices] * signal_ratio,
            float(a),
        )

    # If we are looking at QCD cut, then signal vs BIB roc curve
    # Use roc_curve function from scikit-learn
    if label == 0:
        y_roc = label_binarize(y_left, classes=[0, 1, 2])
        (fpr, tpr, _) = roc_curve(
            y_roc[:, 1],  # type: ignore
            prediction_left_signal,
            sample_weight=weight_left,
            pos_label=1,
        )
        # Scale results by bib_ratio, signal_ratio
        a = auc((1 - fpr) * bib_ratio, tpr * signal_ratio)

        # return results of roc curve
        return (1 / fpr) * bib_ratio, tpr * signal_ratio, float(a)

    raise ValueError("label must be 0 or 2")
