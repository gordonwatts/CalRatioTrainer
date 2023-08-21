import logging
from typing import Optional, Tuple, cast

import numpy as np
import pandas as pd

from cal_ratio_trainer.config import TrainingConfig


def prepare_training_datasets(
    df: pd.DataFrame, df_adversary: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    pd.Series,
    pd.Series,
    float,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Sets up dataframes into X (input variables), Y (labels), weights (jet weights),
    Z (Extra info)

    :param df: main jet dataframe
    :param df_adversary: CR jet dataframe
    :return: X,Y,Z, main jets and adversary
    """
    Y = df["label"]
    Y_adversary = df_adversary["label"].copy()
    Y_adversary.loc[Y_adversary == 2] = 1
    Y_adversary = np.array(Y_adversary.values)

    # Pull out the weights for later use
    weights = df["mcEventWeight"]
    weights_adversary = df_adversary["mcEventWeight"]
    mcWeights = df["mcEventWeight"].copy()
    mcWeights_adversary = df_adversary["mcEventWeight"]

    # Rescale the weights so that signal and qcd have the same weight.
    # TODO: Adversary rescaling is done in utils.py, in match_adversary_weights.
    #       Make this symmetric so re-scaling is done in the "same" place.
    qcd_weight = cast(float, np.sum(mcWeights[Y == 0]))
    sig_weight = cast(float, np.sum(mcWeights[Y == 1]))

    logging.debug("mcWeights")
    logging.debug("label 0: " + str(mcWeights[Y == 0]))

    mcWeights.loc[Y == 0] = mcWeights[Y == 0] * (sig_weight / qcd_weight)

    # TODO: Why not rescale BIB so it also has the same weight?

    # Hard code start and end of names of variables
    X = df.loc[:, "clus_pt_0":"nn_MSeg_t0_29"]
    X = df.loc[:, "jet_pt":"jet_phi"].join(X)
    X["eventNumber"] = df["eventNumber"]

    X_adversary = df_adversary.loc[:, "clus_pt_0":"MSeg_t0_29"]
    X_adversary = df_adversary.loc[:, "jet_pT":"jet_phi"].join(X_adversary)
    X_adversary["eventNumber"] = df_adversary["eventNumber"]

    # Label Z as parametrized variables
    Z = df.loc[:, "llp_mH":"llp_mS"]
    Z_adversary = df_adversary.loc[:, "jet_pT":"jet_eta"]

    return (
        X,
        X_adversary,
        Y,
        Y_adversary,
        mcWeights,
        mcWeights_adversary,
        sig_weight,
        weights,
        weights_adversary,
        Z,
        Z_adversary,
    )


class evaluationObject:
    def __init__(self):
        self.training_params: Optional[TrainingConfig] = None

        # Values we track
        self.all_significance = -1
        self.all_auc = -1
        self.all_aus = -1
        self.qcd_significance = -1
        self.qcd_aus = -1
        self.bib_significance = -1
        self.bib_aus = -1
        self.mh60_significance = -1
        self.mh60_auc = -1
        self.mh60_aus = -1
        self.mh125_significance = -1
        self.mh125_auc = -1
        self.mh125_aus = -1
        self.mh200_significance = -1
        self.mh200_auc = -1
        self.mh200_aus = -1
        self.mh400_significance = -1
        self.mh400_auc = -1
        self.mh400_aus = -1
        self.mh600_significance = -1
        self.mh600_auc = -1
        self.mh600_aus = -1
        self.mh1000_significance = -1
        self.mh1000_auc = -1
        self.mh1000_aus = -1

    def fillObject_sOverB(self, label, value):
        if "600" in label:
            self.mh600_significance = value
        elif "60" in label:
            self.mh60_significance = value
        elif "125" in label:
            self.mh125_significance = value
        elif "200" in label:
            self.mh200_significance = value
        elif "400" in label:
            self.mh400_significance = value
        elif "1000" in label:
            self.mh1000_significance = value
        elif "BIB" in label:
            self.bib_significance = value
        elif "QCD" in label:
            self.qcd_significance = value
        else:
            self.all_significance = value

    def fillObject_auc(self, label, value):
        if "600" in label:
            self.mh600_auc = value
        elif "60" in label:
            self.mh60_auc = value
        elif "125" in label:
            self.mh125_auc = value
        elif "200" in label:
            self.mh200_auc = value
        elif "400" in label:
            self.mh400_auc = value
        elif "600" in label:
            self.mh600_auc = value
        elif "1000" in label:
            self.mh1000_auc = value
        else:
            self.all_auc = value

    def fillObject_aus(self, label, value):
        if "600" in label:
            self.mh600_aus = value
        elif "60" in label:
            self.mh60_aus = value
        elif "125" in label:
            self.mh125_aus = value
        elif "200" in label:
            self.mh200_aus = value
        elif "400" in label:
            self.mh400_aus = value
        elif "600" in label:
            self.mh600_aus = value
        elif "1000" in label:
            self.mh1000_aus = value
        elif "BIB" in label:
            self.bib_aus = value
        elif "QCD" in label:
            self.qcd_aus = value
        else:
            self.all_aus = value

    def fillObject_params(self, training_params: TrainingConfig):
        self.training_params = training_params
