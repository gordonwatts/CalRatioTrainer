import logging
from io import TextIOWrapper
from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
