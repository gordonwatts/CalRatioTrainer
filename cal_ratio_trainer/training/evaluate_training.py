# Make signal, bib, qcd weight plots
import logging
import math
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import atlas_mpl_style as ampl
import numpy as np
import pandas as pd
from keras import Model
from keras.src.utils import np_utils
from matplotlib import pyplot as plt
from cal_ratio_trainer.common.evaulation import (
    normalize_to_one,
    plot_roc_curve,
    signal_llp_efficiencies,
)

from cal_ratio_trainer.training.training_utils import evaluationObject

# Make ATLAS plots
ampl.use_atlas_style()


def ks_w2(
    data1: np.ndarray, data2: np.ndarray, wei1: np.ndarray, wei2: np.ndarray
) -> np.ndarray:
    """To calculate weighted KS

    :param data1: one of the two distributions to use in KS test
    :param data2: other distribution to use in KS test
    :param wei1: Weights of first distribution
    :param wei2: Weights of second distribution
    :return:
    """
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])  # type: ignore
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])  # type: ignore
    cdf1we = cwei1[[np.searchsorted(data1, data, side="right")]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side="right")]]

    return np.max(np.abs(cdf1we - cdf2we))


def plot_prediction_histograms(
    destination: Path,
    prediction: np.ndarray,
    labels: Union[pd.Series, np.ndarray],
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
    do_ks: bool = False,
) -> Optional[
    Tuple[Union[np.ndarray, int], Union[np.ndarray, int], Union[np.ndarray, int]]
]:
    # Get the plots setup
    sig_rows = np.where(labels == 1)
    bkg_rows = np.where(labels == 0)
    bib_rows = np.where(labels == 2)

    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""
    sig_string = "Signal"
    if len(bib_rows[0]) == 0:
        sig_string = "Data"

    weight[bkg_rows] = weight[bkg_rows] / np.sum(weight[bkg_rows])
    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])

    fig, ax = plt.subplots()
    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"

    bin_list = np.linspace(0, 0.0007, 5)
    bin_list = np.append(bin_list, np.logspace(np.log10(0.001), np.log10(1.0), 50))

    # Plot Signal Prediction
    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 1], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 1], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 1],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")

    plt.yscale("log")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("Signal NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")

    ax.set_xlim(0.00002, 1.0)
    ax.set_ylim(top=2)
    ax.legend(loc="upper right")
    ks_sig = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 1],
            prediction[bkg_rows][:, 1],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_sig = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 1]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 1]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "sig_predictions.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    # Next, do the qcd prediction
    fig, ax = plt.subplots()
    bin_list = np.linspace(0, 0.0007, 5)
    bin_list = np.append(bin_list, np.logspace(np.log10(0.001), np.log10(1.0), 50))

    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 0], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 0], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 0],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0  # type: ignore
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")

    plt.yscale("log")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("SM Multijet NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")
    ax.set_xlim(0.00002, 1.0)
    ax.set_ylim(top=2.0)

    ax.legend(loc="upper right")
    ks_qcd = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 0],
            prediction[bkg_rows][:, 0],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_qcd = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 0]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 0]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "qcd_predictions.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    # Finally, do the bib prediction
    fig, ax = plt.subplots()
    bin_list = np.linspace(0, 0.0007, 5)
    bin_list = np.append(bin_list, np.logspace(np.log10(0.001), np.log10(1.0), 50))
    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 2], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 2], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 2],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")
    plt.yscale("log")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("BIB NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")
    ax.set_xlim(0.00002, 1.0)
    ax.set_ylim(top=2)

    plt.xscale("log")
    ax.legend(loc="upper right")
    ks_bib = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 2],
            prediction[bkg_rows][:, 2],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_bib = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 2]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 2]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "bib_predictions.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)
    if do_ks:
        return ks_qcd, ks_sig, ks_bib
    else:
        return


def plot_prediction_histograms_linear(
    destination: Path,
    prediction: np.ndarray,
    labels: pd.Series,
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
    do_ks: bool = False,
):
    # TODO: This looks like identical (almost) code to other plot
    # prediction histogram methods, could we combine with a minor change?
    sig_rows = np.where(labels == 1)
    bkg_rows = np.where(labels == 0)
    bib_rows = np.where(labels == 2)
    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""
    sig_string = "Signal"
    if len(bib_rows[0]) == 0:
        sig_string = "Data"
    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"
    weight[bkg_rows] = weight[bkg_rows] / np.sum(weight[bkg_rows])
    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])

    fig, ax = plt.subplots()

    # bin_list = np.zeros(1)
    bin_list = np.linspace(0.0, 1.0, 30)

    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 1], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 1], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 1],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("Signal NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")

    ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.1, 500000)
    ax.legend(loc="upper right")
    ks_sig = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 1],
            prediction[bkg_rows][:, 1],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_sig = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 1]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 1]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "sig_predictions_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    fig, ax = plt.subplots()
    # bin_list = np.zeros(1)
    bin_list = np.linspace(0.0, 1.0, 30)

    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 0], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 0], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 0],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("SM Multijet NN score", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")
    ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.1, 500000)
    ax.legend(loc="upper right")
    ks_qcd = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 0],
            prediction[bkg_rows][:, 0],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_qcd = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 0]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 0]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "qcd_predictions_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    fig, ax = plt.subplots()
    bin_list = np.linspace(0, 1.0, 30)
    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 2], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 2], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 2],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC")
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB")
    ampl.plot.draw_atlas_label(
        0.02, 0.975, ax=ax, status="int", energy="13 TeV", lumi=139.0, desc=desc_phrase
    )
    ax.set_xlabel("BIB NN weight", loc="right")
    ax.set_ylabel("Fraction of Events", loc="top")
    ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.1, 500000)
    ax.legend(loc="upper right")
    ks_bib = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 2],
            prediction[bkg_rows][:, 2],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_bib = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 2]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 2]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "bib_predictions_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)
    if do_ks:
        return ks_qcd, ks_sig, ks_bib
    else:
        return


def plot_prediction_histograms_halfLinear(
    destination: Path,
    prediction: np.ndarray,
    labels: pd.Series,
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
    do_ks: bool = False,
):
    """semi-Log prediction for BIB, QCD, Signal weight distributions - for paper plots

    :param destination: where to save
    :param prediction: jet NN output
    :param labels: label of jets
    :param weight: jet weights
    :param extra_string: to add to plot
    :param do_ks: if we should calculate KS divergence (for CR jets only)
    :return:
    """
    sig_rows = np.where(labels == 1)
    bkg_rows = np.where(labels == 0)
    bib_rows = np.where(labels == 2)
    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""
    sig_string = "Signal"
    if len(bib_rows[0]) == 0:
        sig_string = "Data"

    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"
    weight[bkg_rows] = weight[bkg_rows] / np.sum(weight[bkg_rows])
    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])

    fig, ax = plt.subplots()

    # bin_list = np.zeros(1)
    bin_list = np.linspace(0.0, 1.0, 30)

    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 1], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 1], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 1],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC", ms=10)
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB", ms=10)
    ampl.plot.draw_atlas_label(
        0.02,
        0.95,
        ax=ax,
        status="int",
        energy="13 TeV",
        lumi=139.0,
        desc=desc_phrase,
        fontsize=25,
    )
    ax.set_xlabel("Signal NN score", loc="right", fontsize=25)
    ax.set_ylabel("Fraction of Events", loc="top", fontsize=25)
    ax.tick_params(labelsize=25)
    plt.yscale("log")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(top=50.0)
    ax.legend(loc="upper right", fontsize=25)
    ks_sig = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 1],
            prediction[bkg_rows][:, 1],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_sig = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 1]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 1]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "sig_predictions_half_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    fig, ax = plt.subplots()
    # bin_list = np.zeros(1)
    bin_list = np.linspace(0.0, 1.0, 30)

    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 0], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 0], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 0],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC", ms=10)
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB", ms=10)
    ampl.plot.draw_atlas_label(
        0.02,
        0.95,
        ax=ax,
        status="int",
        energy="13 TeV",
        lumi=139.0,
        desc=desc_phrase,
        fontsize=25,
    )
    ax.set_xlabel("SM Multijet NN score", loc="right", fontsize=25)
    ax.set_ylabel("Fraction of Events", loc="top", fontsize=25)
    plt.yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=25)
    ax.set_ylim(top=50.0)
    ax.legend(loc="upper right", fontsize=25)
    ks_qcd = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 0],
            prediction[bkg_rows][:, 0],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_qcd = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 0]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 0]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "qcd_predictions_half_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)

    fig, ax = plt.subplots()
    bin_list = np.linspace(0, 1.0, 30)
    n_qcd, bin_edges_qcd = np.histogram(
        prediction[bkg_rows][:, 2], weights=weight[bkg_rows], bins=bin_list
    )
    n_bib, bin_edges_bib = np.histogram(
        prediction[bib_rows][:, 2], weights=weight[bib_rows], bins=bin_list
    )
    ax.hist(
        prediction[sig_rows][:, 2],
        weights=weight[sig_rows],
        color="red",
        alpha=0.5,
        linewidth=0,
        histtype="stepfilled",
        bins=bin_list,
        label=sig_string,
    )
    bin_centers_qcd = (bin_edges_qcd[:-1] + bin_edges_qcd[1:]) / 2.0
    ax.errorbar(bin_centers_qcd, n_qcd, fmt="ok", label="SM Multijet MC", ms=10)
    bin_centers_bib = (bin_edges_bib[:-1] + bin_edges_bib[1:]) / 2.0
    ax.errorbar(bin_centers_bib, n_bib, fmt="ok", mfc="none", label="BIB", ms=10)
    ampl.plot.draw_atlas_label(
        0.02,
        0.95,
        ax=ax,
        status="int",
        energy="13 TeV",
        lumi=139.0,
        desc=desc_phrase,
        fontsize=25,
    )
    ax.set_xlabel("BIB NN score", loc="right", fontsize=25)
    ax.set_ylabel("Fraction of Events", loc="top", fontsize=25)
    plt.yscale("log")
    ax.tick_params(labelsize=25)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(top=200.0)
    ax.legend(loc="upper right", fontsize=25)
    ks_bib = -1
    if do_ks:
        ks_div = ks_w2(
            prediction[sig_rows][:, 2],
            prediction[bkg_rows][:, 2],
            weight[sig_rows],
            weight[bkg_rows],
        )
        ks_bib = ks_div
        reject = ks_div > 2 * np.sqrt(
            (len(prediction[sig_rows][:, 2]) + np.sum(weight[bkg_rows]))  # type: ignore
            / (
                len(prediction[sig_rows][:, 2]) * np.sum(weight[bkg_rows])
            )  # type: ignore
        )
        plt.text(
            0.85,
            0.9,
            f"KS: {ks_div:.4f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.text(
            0.85,
            0.8,
            f"Reject: {reject}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.savefig(
        destination / (extra_string + "bib_predictions_half_linear.png"),
        format="png",
        transparent=True,
    )
    plt.clf()
    plt.close(fig)
    if do_ks:
        return ks_qcd, ks_sig, ks_bib
    else:
        return


def bkg_falsePositives(
    prediction: np.ndarray,
    y_test: pd.Series,
    Z_test: pd.DataFrame,
    destination: Path,
    f: TextIOWrapper,
):
    """Makes plot of false positives for signal

    :param prediction: jet NN output
    :param y_test: jet labels
    :param Z_test: signal mH, mS
    :param destination: where to save file
    :param f:
    """
    qcd_rows = np.where(y_test == 0)
    bib_rows = np.where(y_test == 2)
    bkg_rows = [np.concatenate((qcd_rows[0], bib_rows[0]))]

    prediction = prediction[bkg_rows[0]]
    Z_test = Z_test.iloc[bkg_rows[0]]
    mass_array = (
        Z_test.groupby(["llp_mH", "llp_mS"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    plot_x = []
    plot_y = []
    plot_z = []

    # Loop over signal mH, mS
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
        logging.info(
            "mH: " + str(mH) + ", mS: " + str(mS) + ", False positive: " + str(temp_eff)
        )
        f.write("%s,%s,%s\n" % (str(mH), str(mS), str(temp_eff)))

    plt.clf()
    plt.figure()
    plt.scatter(
        plot_x,
        plot_y,
        marker="+",  # type: ignore
        s=150,
        linewidths=4,
        c=plot_z,
        cmap=plt.cm.coolwarm,  # type: ignore
    )
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"mS")
    plt.xlabel("mH")
    plt.ylabel("False Positive Rate")

    plt.savefig(destination / "bkg_falsePositives.png", format="png", transparent=True)
    plt.clf()


def do_checkpoint_prediction_histogram(
    model: Model,
    dir_name: Path,
    adv_x: List[np.ndarray],
    adv_y: np.ndarray,
    adv_weights: np.ndarray,
    plot_name_prefix: str,
    high_mass: bool,
    low_mass: bool,
) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray], Union[int, np.ndarray]]:
    """Makes histogram plots for adversarial training

    :param model: name, to add to filename
    :param dir_name: where to save
    :param adv_x: adversarial jet NN input variables
    :param adv_y: adversarial labels
    :param adv_weights: adversarial jet weights
    :param epoch: which epoch
    :return:
    """

    validation_prediction = model.predict(adv_x, verbose=0)  # type: ignore

    jet_pt = adv_x[3]
    jet_pt = jet_pt[:, 0]

    # TODO: This first call to the prediction isn't having its results used.
    # Can we just delete it?
    r = plot_prediction_histograms(
        dir_name,
        validation_prediction,
        adv_y,
        adv_weights,
        f"{plot_name_prefix}_adversary",
        high_mass,
        low_mass,
        do_ks=True,
    )
    assert r is not None
    ks_qcd, ks_sig, ks_bib = r

    lowPt_validation_prediction = validation_prediction[jet_pt < 0.25, :]
    lowPt_adv_y = adv_y[jet_pt < 0.25]
    lowPt_adv_weights = adv_weights[jet_pt < 0.25]
    r = plot_prediction_histograms(
        dir_name,
        lowPt_validation_prediction,
        lowPt_adv_y,
        lowPt_adv_weights,
        f"{plot_name_prefix}_adversary_lowPt",
        high_mass,
        low_mass,
        do_ks=True,
    )
    assert r is not None
    ks_qcd_low, ks_sig_low, ks_bib_low = r

    midPt_validation_prediction = validation_prediction[
        (jet_pt > 0.25) & (jet_pt < 0.5), :
    ]

    midPt_adv_y = adv_y[(jet_pt > 0.25) & (jet_pt < 0.5)]
    midPt_adv_weights = adv_weights[(jet_pt > 0.25) & (jet_pt < 0.5)]
    r = plot_prediction_histograms(
        dir_name,
        midPt_validation_prediction,
        midPt_adv_y,
        midPt_adv_weights,
        f"{plot_name_prefix}_adversary_midPt",
        high_mass,
        low_mass,
        do_ks=True,
    )
    assert r is not None
    ks_qcd_mid, ks_sig_mid, ks_bib_mid = r

    highPt_validation_prediction = validation_prediction[jet_pt > 0.5, :]
    highPt_adv_y = adv_y[jet_pt > 0.5]
    highPt_adv_weights = adv_weights[jet_pt > 0.5]
    r = plot_prediction_histograms(
        dir_name,
        highPt_validation_prediction,
        highPt_adv_y,
        highPt_adv_weights,
        f"{plot_name_prefix}_adversary_highPt",
        high_mass,
        low_mass,
        do_ks=True,
    )
    assert r is not None
    ks_qcd_high, ks_sig_high, ks_bib_high = r

    ks_qcd = ks_qcd_low + ks_qcd_mid + ks_qcd_high
    ks_sig = ks_sig_low + ks_sig_mid + ks_sig_high
    ks_bib = ks_bib_low + ks_bib_mid + ks_bib_high
    return ks_qcd, ks_sig, ks_bib


def checkpoint_pred_hist_main(
    model: Model,
    dir_name: Path,
    x_test: List[np.ndarray],
    y_test: pd.Series,
    mcWeights_test: pd.Series,
    epoch: int,
    high_mass: bool,
    low_mass: bool,
) -> None:
    validation_prediction = model.predict(x_test, verbose=0)  # type: ignore
    assert isinstance(validation_prediction, np.ndarray)
    assert isinstance(mcWeights_test.values, np.ndarray)

    plot_prediction_histograms_linear(
        dir_name,
        validation_prediction,
        y_test,
        mcWeights_test.values,
        f"{epoch:03d}_main",
        high_mass,
        low_mass,
    )


def print_history_plots(
    adv_loss: List[float],
    adv_acc: List[float],
    val_adv_loss: List[float],
    val_adv_acc: List[float],
    original_lossf: List[float],
    original_acc: List[float],
    val_original_lossf: List[float],
    val_original_acc: List[float],
    original_adv_lossf: List[float],
    original_adv_acc: List[float],
    val_original_adv_lossf: List[float],
    val_original_adv_acc: List[float],
    ks_qcd_hist: List[float],
    ks_sig_hist: List[float],
    ks_bib_hist: List[float],
    dir_name: Path,
):
    """Makes plots like loss, KS divergence, etc... every epoch - for adversary training

    :param advw_array: adversary weight array
    :param adv_loss: adversary loss array
    :param adv_acc: adversary accuracy array
    :param val_adv_loss: adversary loss for validation set
    :param val_adv_acc: adversary accuracy for validation set
    :param original_lossf: loss function for main NN
    :param original_acc: accuracy for main NN
    :param val_original_lossf: loss function for Main NN, validation set
    :param val_original_acc: accuracy for Main NN, validation set
    :param original_adv_lossf:
    :param original_adv_acc:
    :param val_original_adv_lossf:
    :param val_original_adv_acc:
    :param lr_array: learning rate history
    :param ks_qcd_hist: KS divergence for QCD score hist
    :param ks_sig_hist: KS divergence Signal score hist
    :param ks_bib_hist: KS divergence BIB score hist
    :param accept_epoch_array: If KS goes down, accept marked as true
    :param dir_name: where to save plots
    """
    # Plot training & validation accuracy values
    logging.debug("Plotting Adversary History plots...")

    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(adv_loss, label="Train", color="blue")
    plt.title("Train Adversary Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "train_adv_loss.png", format="png", transparent=True)

    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(val_adv_loss, label="Test", color="orange")
    plt.title("Test Adversary Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "test_adv_loss.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(adv_acc, label="Train", color="blue")
    plt.plot(val_adv_acc, label="Test", color="orange")
    plt.title("Adversary Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "adv_acc.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(original_lossf, label="Train", color="blue")
    plt.plot(val_original_lossf, label="Test", color="orange")
    plt.title("Main Network Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "main_nn_loss.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(original_acc, label="Train", color="blue")
    plt.plot(val_original_acc, label="Test", color="orange")
    plt.title("Main Network Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "main_network_acc.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(original_adv_lossf, label="Train", color="blue")
    plt.title("Train Main Adversary Loss function")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "train_main_adv_loss.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(val_original_adv_lossf, label="Test", color="orange")
    plt.title("Test Main Adversary Loss function")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "test_main_adv_loss.png", format="png", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(original_adv_acc, label="Train", color="blue")
    plt.plot(val_original_adv_acc, label="Test", color="orange")
    plt.title("Adversary only Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "main_adv_acc.png", format="png", transparent=True)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(ks_qcd_hist, label="KS QCD", color="blue")
    plt.title("KS test for QCD score")
    plt.ylabel("KS Divergence")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "ks_qcd.png", format="png", transparent=True)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(ks_sig_hist, label="KS Signal", color="blue")
    plt.title("KS test for Signal score")
    plt.ylabel("KS Divergence")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "ks_sig.png", format="png", transparent=True)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(ks_bib_hist, label="KS BIB", color="blue")
    plt.title("KS test for BIB score")
    plt.ylabel("KS Divergence")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(dir_name / "ks_bib.png", format="png", transparent=True)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.close("all")


def evaluate_model(
    model: Model,
    discriminator_model: Model,
    dir_name: Path,
    X_test: List[np.ndarray],
    y_test: pd.Series,
    weights_test: List[np.ndarray],
    Z_test: pd.DataFrame,
    mcWeights_test: pd.Series,
    X_test_adv: List[np.ndarray],
    y_test_adv: np.ndarray,
    weights_test_adversary: pd.Series,
    n_folds: Optional[int],
    eval_object: evaluationObject,
    Z_test_adversary: pd.DataFrame,
    high_mass: bool,
    low_mass: bool,
) -> Tuple[float, float]:
    """Where we start making plots, AUC calcs when training is done

    :param model: the main NN model
    :param discriminator_model: adversary model
    :param dir_name: where to save plots
    :param X_test: input variables of the testing set
    :param y_test: labels of testing set
    :param weights_test: jet weights of testing set (flattened + MC)
    :param Z_test: mH, mS of testing set
    :param mcWeights_test: jet weights of testing set (MC, not flattened)
    :param X_test_adv: input variables of CR jets
    :param y_test_adv: labels of CR jets
    :param weights_test_adversary: weights of CR jets
    :param n_folds: which kfold
    :param eval_object: where to save some training variables for later study
    :param Z_test_adversary: extra variables for CR
    :param skipTraining: if we skipped training
    :return: roc curves
    """
    # TODO: Many of these plots are already made on a per-epoch basis.
    # Can we just save those instead of making them again here?

    # evaluate the model using Keras api
    # model.evaluate expects target data to be the same shape/format as model.fit
    y_eval = np_utils.to_categorical(y_test)
    y_eval = [y_eval, y_eval, y_eval, y_eval, y_eval]

    # TODO: refactor and understand
    # make predictions
    # currently expects X to be a list of length 4 (model has 4 inputs)
    validation_prediction = model.predict(X_test, verbose=0)  # type: ignore
    # model currently has 5 outputs (1 main output, 4 outputs for monitoring LSTMs)
    prediction = validation_prediction

    # TODO: Go through and change these `verbose=0` to whatever they should be.
    validation_prediction_adv = model.predict(X_test_adv, verbose=0)  # type: ignore

    # TODO: Make sure these types are right!
    plot_prediction_histograms(
        dir_name,
        prediction,
        y_test,
        mcWeights_test.values,  # type: ignore
        "",
        high_mass,
        low_mass,
    )
    plot_prediction_histograms_linear(
        dir_name,
        prediction,
        y_test,
        mcWeights_test.values,  # type: ignore
        "",
        high_mass,
        low_mass,
    )
    plot_prediction_histograms_halfLinear(
        dir_name,
        prediction,
        y_test,
        mcWeights_test.values,  # type: ignore
        "",
        high_mass,
        low_mass,
    )
    plot_prediction_histograms(
        dir_name,
        validation_prediction_adv,
        y_test_adv,
        weights_test_adversary.values,  # type: ignore
        "adv",
        high_mass,
        low_mass,
    )

    # Sum of MC weights
    mcWeights_test = normalize_to_one(mcWeights_test, y_test)  # type: ignore

    # This will be the BIB efficiency to aim for when making ROC curve
    # threshold = 1 - 0.0316
    threshold = 0
    # Third label: the label of the class we are doing a 'family' of. Other two
    # classes will make the ROC curve
    third_label = 2
    # We'll be writing the stats to training_details.txt
    with (dir_name / "training_details.txt").open("a") as f:
        if n_folds:
            f.write("\nKFold iteration # %s" % str(n_folds))
        f.write("\nEvaluation metrics\n")

        # Find threshold, or at what label we will have the required
        # percentage of 'test_label' correctly predicted
        # Make plots of signal efficiency vs mH, mS
        fig = signal_llp_efficiencies(prediction, y_test, Z_test, f)[0]
        fig.savefig(
            str(dir_name / "signal_llp_efficiencies.png"),
            format="png",
            transparent=True,
        )
        plt.close(fig)

        bkg_falsePositives(prediction, y_test, Z_test, dir_name, f)

        max_SoverB, roc_auc = setup_separate_evaluations(
            prediction,
            y_test,
            mcWeights_test,
            Z_test,
            third_label,
            threshold,
            dir_name,
            f,  # type: ignore
            eval_object,
            n_folds,
        )

    return roc_auc, max_SoverB


def setup_separate_evaluations(
    prediction: np.ndarray,
    y_test: pd.Series,
    mcWeights_test: pd.Series,
    Z_test: pd.DataFrame,
    third_label: int,
    threshold: int,
    destination: Path,
    f: TextIOWrapper,
    eval_object: evaluationObject,
    n_folds: Optional[int],
) -> Tuple[float, float]:
    """Does all the output plot + calculations separately for different mH

    :param prediction: output of jet NN
    :param y_test: actual labels
    :param mcWeights_test: MC weights
    :param Z_test: mH, mS of signal
    :param third_label: for ROC curves
    :param threshold: for ROC curves
    :param destination: where to save
    :param f:
    :param eval_object: to save in eval_object, for later study
    :param n_folds: how many folds in KFold
    :return:
    """
    max_SoverB = significance_scan(
        prediction, y_test, mcWeights_test, destination, f, "", eval_object, n_folds
    )
    roc_auc, fig = plot_roc_curve(
        f,
        mcWeights_test,  # type: ignore
        prediction,
        third_label,
        threshold,
        y_test.values,  # type: ignore
        "",
    )
    result = (max_SoverB, roc_auc)
    eval_object.fillObject_auc("", roc_auc)
    plot_name_stub = (
        "roc_curve_atlas_rej_bib_" if third_label == 2 else "roc_curve_atlas_rej_qcd_"
    )
    fig.savefig(
        str(destination / f"{plot_name_stub}_{n_folds}.png"),
        format="png",
        transparent=True,
    )
    plt.close(fig)

    sig_rows = np.where(y_test == 1)
    bib_rows = np.where(y_test == 2)
    qcd_rows = np.where(y_test == 0)

    mass_array = (
        Z_test.groupby(["llp_mH"]).size().reset_index().rename(columns={0: "count"})
    )

    sig_pred = prediction[sig_rows[0]]
    sig_y_test = y_test.values[sig_rows[0]]
    sig_Z_test = Z_test.iloc[sig_rows[0]]
    sig_mcWeights_test = mcWeights_test.iloc[sig_rows[0]]

    qcd_pred = prediction[qcd_rows[0]]
    qcd_y_test = y_test.values[qcd_rows[0]]
    # qcd_Z_test = Z_test.iloc[qcd_rows[0]]
    qcd_mcWeights_test = mcWeights_test.iloc[qcd_rows[0]]

    bib_pred = prediction[bib_rows[0]]
    bib_y_test = y_test.values[bib_rows[0]]
    # bib_Z_test = Z_test.iloc[bib_rows[0]]
    bib_mcWeights_test = mcWeights_test.iloc[bib_rows[0]]

    temp_pred = np.concatenate((sig_pred, qcd_pred))
    temp_y_test = np.concatenate((sig_y_test, qcd_y_test))
    temp_mcWeights_test = np.concatenate((sig_mcWeights_test, qcd_mcWeights_test))

    significance_scan(
        temp_pred,
        temp_y_test,
        temp_mcWeights_test,
        destination,
        f,
        "QCD only",
        eval_object,
        n_folds,
    )

    temp_pred = np.concatenate((sig_pred, bib_pred))
    temp_y_test = np.concatenate((sig_y_test, bib_y_test))
    temp_mcWeights_test = np.concatenate((sig_mcWeights_test, bib_mcWeights_test))

    significance_scan(
        temp_pred,
        temp_y_test,
        temp_mcWeights_test,
        destination,
        f,
        "BIB only",
        eval_object,
        n_folds,
    )

    for item, mH in zip(mass_array["count"], mass_array["llp_mH"]):
        temp_sig_pred = sig_pred[(sig_Z_test["llp_mH"] == mH).values]
        temp_sig_y_test = sig_y_test[(sig_Z_test["llp_mH"] == mH).values]
        temp_sig_mcWeights_test = sig_mcWeights_test[
            (sig_Z_test["llp_mH"] == mH).values
        ]

        temp_pred = np.concatenate((temp_sig_pred, qcd_pred, bib_pred))
        temp_y_test = np.concatenate((temp_sig_y_test, qcd_y_test, bib_y_test))
        temp_mcWeights_test = np.concatenate(
            (temp_sig_mcWeights_test, qcd_mcWeights_test, bib_mcWeights_test)
        )

        label_string = f"mH{mH}"
        significance_scan(
            temp_pred,
            temp_y_test,
            temp_mcWeights_test,
            destination,
            f,
            label_string,
            eval_object,
            n_folds,
        )
        roc_auc, fig = plot_roc_curve(
            f,
            temp_mcWeights_test,
            temp_pred,
            third_label,
            threshold,
            temp_y_test,
            label_string,
        )
        eval_object.fillObject_auc(label_string, roc_auc)
        plot_name_stub = (
            "roc_curve_atlas_rej_bib_"
            if third_label == 2
            else "roc_curve_atlas_rej_qcd_"
        )
        fig.savefig(
            str(destination / f"{plot_name_stub}_{n_folds}.png"),
            format="png",
            transparent=True,
        )
        plt.close(fig)

    return result


def significance_scan(
    prediction: np.ndarray,
    y_test: Union[pd.Series, np.ndarray],
    weight: Union[pd.Series, np.ndarray],
    destination: Path,
    f: TextIOWrapper,
    label_string: str,
    eval_object: evaluationObject,
    n_folds: Optional[int],
) -> float:
    """Scans significance in NN output value cuts

    :param prediction: NN output
    :param y_test: labels
    :param weight: jet weights
    :param destination: where to save file
    :param f:
    :param label_string: how to label figure
    :param eval_object: object which saves parameters for further study
    :param n_folds: how many kfold's
    :return:
    """
    start_of_scan = np.logspace(-5, -1, 100)
    middle_of_scan = np.linspace(0.11, 0.89, 20)
    end_of_scan = np.logspace(0, -0.04575, 600)

    thresholds = np.concatenate((start_of_scan, middle_of_scan, end_of_scan[::-1]))

    qcd_rows = np.where(y_test == 0)
    bib_rows = np.where(y_test == 2)
    sig_rows = np.where(y_test == 1)

    SoverB_array = []
    sig_prediction = prediction[:, 1]
    tot_bib = float(len(bib_rows[0]))
    weight = np.array(weight)
    qcd_sum = cast(float, np.sum(weight[qcd_rows[0]]))
    correction_factor = 1
    if tot_bib > 0 and qcd_sum > 0:
        correction_factor = tot_bib / qcd_sum

    counter = 0
    termination = 0
    for threshold in thresholds:
        num_sig = len(np.where((sig_prediction[sig_rows[0]]) > threshold)[0])
        num_bib = len(np.where(sig_prediction[bib_rows[0]] > threshold)[0])
        temp_qcd = sig_prediction[qcd_rows[0]] > threshold
        temp_weight = weight[qcd_rows[0]]
        temp_qcd = temp_weight[temp_qcd]
        num_qcd = cast(float, np.sum(temp_qcd))
        num_bkg = cast(float, num_bib + (num_qcd * correction_factor))

        if num_bkg > 10:
            SoverB_array.append(num_sig / (math.sqrt(num_bkg)))
            termination = counter
        else:
            break
        counter = counter + 1

    plt.plot(1 - thresholds[: termination + 1], SoverB_array, label=str(label_string))
    plt.xlabel("1-(signal weight threshold)")
    plt.ylabel("S over Root B")
    plt.xscale("log")

    plt.savefig(
        destination
        / ("SoverB_scan_" + str(label_string) + "_" + str(n_folds) + ".png"),
        format="png",
        transparent=True,
    )
    plt.clf()

    area_under_significance = np.trapz(SoverB_array, thresholds[: termination + 1])

    f.write("AuS: %s, label: %s\n" % (str(area_under_significance), str(label_string)))
    f.write(
        "S over Root B: %s, label: %s\n"
        % (str(np.amax(SoverB_array)), str(label_string))
    )
    eval_object.fillObject_sOverB(str(label_string), np.amax(SoverB_array))
    eval_object.fillObject_aus(str(label_string), area_under_significance)

    return cast(float, np.amax(SoverB_array))
