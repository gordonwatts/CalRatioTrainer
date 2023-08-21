import logging
from typing import List, Tuple

import pandas as pd
import numpy as np


class ModelInput:
    """
    This is a class for inputs to the model i.e. constit, track, MSeg, jet (subclass)

    Attributes:
        name (str): name of input
        rows_max (int): number of inputs (number of rows in input data variable X)
        num_features (int): number of features or variables (number of columns in input
        data variable X) filters_cnn (list): list of number of filters for each Conv1D
        layer nodes_lstm (int): number of nodes for lstm layer
    """

    def __init__(
        self,
        name: str,
        rows_max: int,
        num_features: int,
        filters_cnn: List[int] = [],
        nodes_lstm: int = 0,
        lstm_layers: int = 1,
        mH_mS_parametrization: List[bool] = [False, False],
    ):
        self.name = name
        self.rows_max = rows_max
        self.num_features = num_features
        self.filters_cnn = filters_cnn
        self.nodes_lstm = nodes_lstm
        self.lstm_layers = lstm_layers
        self.mH_mS_parametrization = mH_mS_parametrization

    def extract_and_split_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        Z_train: pd.DataFrame,
        Z_val: pd.DataFrame,
        Z_test: pd.DataFrame,
        start: str,
        end: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and splits up the data into training, validation, and testing inputs
        :return: training, validation, and testing variables
        """
        train = X_train.loc[
            :, start : end + str(self.rows_max - 1)  # noqa
        ]  # type: ignore
        filter_Pixel = [col for col in train if "track_Pixel" in str(col)]
        filter_constitTime = [col for col in train if str(col).startswith("clus_time")]
        filter_constitTimeZero = [
            col for col in train if str(col).startswith("clus_time_0")
        ]
        filter_constitTimeOne = [col for col in train if col == "clus_time_1"]
        # TODO: Remove these next lines as it does not look like it is used.
        # A LOT OF THIS CODE APPEARS TO BE DEAD AS YOU COMMENT OUT MORE AND
        # MORE UNUSED. There also seems to be a lot of repetition between
        # the test/train dat split.
        # filter_msegTime = [col for col in train if col.startswith("nn_MSeg_t0")]
        for pixel in filter_Pixel:
            del train[pixel]
        for constit in filter_constitTime:
            # train[constit] = train[filter_constitTimeZero]
            # del train[constit]
            pass
        for constit in filter_constitTimeZero:
            # train[constit] = 0
            # train.loc[train[constit] > 0.2741,constit] = 1
            # train.loc[train[constit] <= 0.2741,constit] = -1
            pass
        for constit in filter_constitTimeOne:
            # train[constit] = 0
            # train.loc[train[constit] > 0.4375,constit] = 1
            # train.loc[train[constit] <= 0.4375,constit] = -1
            pass

        val = X_val.loc[:, start : end + str(self.rows_max - 1)]  # noqa: E203
        filter_Pixel = [col for col in val if "track_Pixel" in str(col)]
        filter_constitTime = [col for col in val if str(col).startswith("clus_time")]
        filter_constitTimeZero = [
            col for col in val if str(col).startswith("clus_time_0")
        ]
        filter_constitTimeOne = [col for col in val if "clus_time_1" == col]
        # filter_msegTime = [col for col in val if col.startswith("nn_MSeg_t0")]
        for pixel in filter_Pixel:
            del val[pixel]
        for constit in filter_constitTime:
            # val[constit] = val[filter_constitTimeZero]
            # del val[constit]
            pass
        for constit in filter_constitTimeZero:
            # val[constit] = 0
            # val.loc[val[constit] > 0.2741,constit] = 1
            # val.loc[val[constit] <= 0.2741,constit] = -1
            pass
        for constit in filter_constitTimeOne:
            # val.loc[val[constit] > 0.4375,constit] = 1
            # val.loc[val[constit] <= 0.4375,constit] = -1
            # val[constit] = 0
            pass
        """
        for mseg in filter_msegTime:
            del val[mseg]
        """
        test = X_test.loc[:, start : end + str(self.rows_max - 1)]  # noqa: E203
        # temp_for_val = test[Z_test['eventNumber'] == 17787]
        filter_Pixel = [col for col in test if "track_Pixel" in str(col)]
        filter_constitTime = [col for col in test if str(col).startswith("clus_time")]
        filter_constitTimeZero = [
            col for col in test if str(col).startswith("clus_time_0")
        ]
        filter_constitTimeOne = [col for col in test if col == "clus_time_1"]
        # filter_msegTime = [col for col in test if col.startswith("nn_MSeg_t0")]
        for pixel in filter_Pixel:
            del test[pixel]
        for constit in filter_constitTime:
            # test[constit] = test[filter_constitTimeZero]
            # del test[constit]
            pass
        for constit in filter_constitTimeZero:
            # test[constit] = 0
            # test.loc[test[constit] > 0.2741,constit] = 1
            # test.loc[test[constit] <= 0.2741,constit] = -1
            pass
        for constit in filter_constitTimeOne:
            # test.loc[test[constit] > 0.4375,constit] = 1
            # test.loc[test[constit] <= 0.4375,constit] = -1
            # test[constit] = 0
            pass
        """
        for mseg in filter_msegTime:
            del test[mseg]
        """
        temp_num_features = self.num_features
        if self.mH_mS_parametrization[0] and self.mH_mS_parametrization[1]:
            index_counter = 0
            for index in range(
                self.num_features,
                (self.num_features * self.rows_max) + 1,
                self.num_features,
            ):
                train.insert(
                    index + (index_counter * 2),
                    "llp_mS_" + str(index_counter),
                    Z_train["llp_mS"],
                )
                test.insert(
                    index + (index_counter * 2),
                    "llp_mS_" + str(index_counter),
                    Z_test["llp_mS"],
                )
                val.insert(
                    index + (index_counter * 2),
                    "llp_mS_" + str(index_counter),
                    Z_val["llp_mS"],
                )
                train.insert(
                    index + (index_counter * 2),
                    "llp_mH_" + str(index_counter),
                    Z_train["llp_mH"],
                )
                test.insert(
                    index + (index_counter * 2),
                    "llp_mH_" + str(index_counter),
                    Z_test["llp_mH"],
                )
                val.insert(
                    index + (index_counter * 2),
                    "llp_mH_" + str(index_counter),
                    Z_val["llp_mH"],
                )
                index_counter = index_counter + 1

            # train = pd.concat([train, Z_train], axis=1)
            # test = pd.concat([test, Z_test], axis=1)
            # val = pd.concat([val, Z_val], axis=1)
            temp_num_features = self.num_features + 2
        elif self.mH_mS_parametrization[0]:
            index_counter = 0
            for index in range(
                self.num_features,
                (self.num_features * self.rows_max) + 1,
                self.num_features,
            ):
                train.insert(
                    index + (index_counter),
                    "llp_mH_" + str(index_counter),
                    Z_train["llp_mH"],
                )
                test.insert(
                    index + (index_counter),
                    "llp_mH_" + str(index_counter),
                    Z_test["llp_mH"],
                )
                val.insert(
                    index + (index_counter),
                    "llp_mH_" + str(index_counter),
                    Z_val["llp_mH"],
                )
                index_counter = index_counter + 1
            temp_num_features = self.num_features + 1
        elif self.mH_mS_parametrization[1]:
            index_counter = 0
            for index in range(
                self.num_features,
                (self.num_features * self.rows_max) + 1,
                self.num_features,
            ):
                train.insert(
                    index + (index_counter),
                    "llp_mS_" + str(index_counter),
                    Z_train["llp_mS"],
                )
                test.insert(
                    index + (index_counter),
                    "llp_mS_" + str(index_counter),
                    Z_test["llp_mS"],
                )
                val.insert(
                    index + (index_counter),
                    "llp_mS_" + str(index_counter),
                    Z_val["llp_mS"],
                )
                index_counter = index_counter + 1
            temp_num_features = self.num_features + 1
        train = train.values.reshape(train.shape[0], self.rows_max, temp_num_features)
        val = val.values.reshape(val.shape[0], self.rows_max, temp_num_features)
        test = test.values.reshape(test.shape[0], self.rows_max, temp_num_features)

        # print some details
        logging.info("Shape: %.0f x %.0f" % (train.shape[1], train.shape[2]))
        logging.info("Number of training examples %.0f" % (train.shape[0]))
        logging.info("Number of validating examples %.0f" % (val.shape[0]))
        logging.info("Number of testing examples %.0f" % (test.shape[0]))

        return train, val, test
