import logging
from typing import Any, List, Tuple

import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Input, Conv1D, GlobalAveragePooling1D
from keras.regularizers import L1L2

from cal_ratio_trainer.config import TrainingConfig


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
        logging.debug("  Shape: %.0f x %.0f" % (train.shape[1], train.shape[2]))
        logging.debug("  Number of training examples %.0f" % (train.shape[0]))
        logging.debug("  Number of validating examples %.0f" % (val.shape[0]))
        logging.debug("  Number of testing examples %.0f" % (test.shape[0]))

        return train, val, test

    def init_keras_layers(
        self,
        shape: Any,
        training_params: TrainingConfig,
        activation_cnn: str = "relu",
        activation_lstm: str = "softmax",
    ):
        """
        Setup the Keras layers for individual ModelInput object
        :return: input, output, and dense tensor variables
        """
        # TODO: Remove the `Any` in the arg list for shape.
        # input to first model layer
        input_tensor_cr = Input(
            shape=shape, dtype="float32", name=self.name + "_input_cr"
        )
        input_tensor_adv = Input(
            shape=shape, dtype="float32", name=self.name + "_input_adv"
        )
        output_tensor = None
        dense_tensor = None

        # check if model input has conv1d layers
        assert training_params.reg_values is not None
        if self.filters_cnn:
            # init output
            output_tensor = Conv1D(
                filters=self.filters_cnn[0],
                kernel_size=1,
                activation=activation_cnn,
                input_shape=shape,
                kernel_regularizer=L1L2(
                    l1=training_params.reg_values, l2=training_params.reg_values
                ),
            )
            output_tensor_cr = output_tensor(input_tensor_cr)
            output_tensor_adv = output_tensor(input_tensor_adv)

            # iterate over conv1d layers
            for filters in self.filters_cnn[1:]:
                # add name to final layer
                if filters == self.filters_cnn[-1]:
                    output_tensor = Conv1D(
                        filters=filters,
                        kernel_size=1,
                        activation=activation_cnn,
                        name=self.name + "_final_conv1d",
                        kernel_regularizer=L1L2(
                            l1=training_params.reg_values, l2=training_params.reg_values
                        ),
                    )
                    output_tensor_cr = output_tensor(output_tensor_cr)
                    output_tensor_adv = output_tensor(output_tensor_adv)
                    output_tensor = Dropout(training_params.dropout_array)
                    output_tensor_cr = output_tensor(output_tensor_cr)
                    output_tensor_adv = output_tensor(output_tensor_adv)
                else:
                    output_tensor = Conv1D(
                        filters=filters,
                        kernel_size=1,
                        activation=activation_cnn,
                        kernel_regularizer=L1L2(
                            l1=training_params.reg_values, l2=training_params.reg_values
                        ),
                    )
                    output_tensor_cr = output_tensor(output_tensor_cr)
                    output_tensor_adv = output_tensor(output_tensor_adv)
                    output_tensor = Dropout(training_params.dropout_array)
                    output_tensor_cr = output_tensor(output_tensor_cr)
                    output_tensor_adv = output_tensor(output_tensor_adv)

            # check if model input has only conv1d layers
            if not self.nodes_lstm:
                output_tensor = GlobalAveragePooling1D()
                output_tensor_cr = output_tensor(output_tensor_cr)
                output_tensor_adv = output_tensor(output_tensor_adv)
        else:
            raise NotImplementedError("Network must have 1D layers")

        # check if model input has an lstm layer
        for i in range(self.lstm_layers):
            if self.nodes_lstm:
                if i == self.lstm_layers - 1:
                    output_tensor = LSTM(
                        self.nodes_lstm,
                        kernel_regularizer=L1L2(
                            l1=training_params.reg_values, l2=training_params.reg_values
                        ),
                    )
                    output_tensor_cr = output_tensor(
                        output_tensor_cr
                        if output_tensor_cr is not None
                        else input_tensor_cr
                    )
                    output_tensor_adv = output_tensor(
                        output_tensor_adv
                        if output_tensor_adv is not None
                        else input_tensor_adv
                    )
                else:
                    output_tensor = LSTM(
                        self.nodes_lstm,
                        return_sequences=True,
                        kernel_regularizer=L1L2(
                            l1=training_params.reg_values, l2=training_params.reg_values
                        ),
                    )
                    output_tensor_cr = output_tensor(
                        output_tensor_cr
                        if output_tensor_cr is not None
                        else input_tensor_cr
                    )
                    output_tensor_adv = output_tensor(
                        output_tensor_adv
                        if output_tensor_adv is not None
                        else input_tensor_adv
                    )

            output_tensor = Dropout(training_params.dropout_array)
            output_tensor_cr = output_tensor(output_tensor_cr)
            output_tensor_adv = output_tensor(output_tensor_adv)

        if not (self.nodes_lstm or self.filters_cnn):
            print("\nNo Conv1D or LSTM layers in model architecture!\n")
            # set output tensor equal to input tensor
            output_tensor_cr = input_tensor_cr
            output_tensor_adv = input_tensor_adv

            raise NotImplementedError("Need either lstm or filters!")

        else:
            # Dense layer to track performance of layer
            dense_tensor = Dense(
                3, activation=activation_lstm, name=self.name + "_output"
            )
            dense_tensor_cr = dense_tensor(output_tensor_cr)
            dense_tensor_adv = dense_tensor(output_tensor_adv)

        return (
            input_tensor_cr,
            output_tensor_cr,
            dense_tensor_cr,
            input_tensor_adv,
            output_tensor_adv,
            dense_tensor_adv,
        )
