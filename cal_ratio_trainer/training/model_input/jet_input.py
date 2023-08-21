from typing import List, Tuple

from cal_ratio_trainer.training.model_input.model_input import ModelInput
import pandas as pd
from keras.layers import Dense, Input


class JetInput(ModelInput):
    """
    This is a subclass for jet input to the model
    """

    def __init__(
        self,
        name: str,
        rows_max: int = 0,
        num_features: int = 0,
        filters_cnn: List[int] = [],
        nodes_lstm: int = 0,
        mH_mS_parametrization: List[bool] = [False, False],
    ):
        ModelInput.__init__(
            self,
            name,
            rows_max,
            num_features,
            filters_cnn,
            nodes_lstm,
            mH_mS_parametrization=mH_mS_parametrization,
        )

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = X_train.loc[:, start:end]
        val = X_val.loc[:, start:end]
        test = X_test.loc[:, start:end]
        if self.mH_mS_parametrization[0] and self.mH_mS_parametrization[1]:
            train = pd.concat([train, Z_train], axis=1)
            test = pd.concat([test, Z_test], axis=1)
            val = pd.concat([val, Z_val], axis=1)
        elif self.mH_mS_parametrization[0]:
            train = pd.concat([train, Z_train["LLP_mH"]], axis=1)
            test = pd.concat([test, Z_test["LLP_mH"]], axis=1)
            val = pd.concat([val, Z_val["LLP_mH"]], axis=1)
        elif self.mH_mS_parametrization[1]:
            train = pd.concat([train, Z_train["LLP_mS"]], axis=1)
            test = pd.concat([test, Z_test["LLP_mS"]], axis=1)
            val = pd.concat([val, Z_val["LLP_mS"]], axis=1)

        # print some details
        print("Shape: %.0f" % (train.shape[1]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of validating examples %.0f" % (val.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))

        return train, val, test

    def init_keras_dense_input_output(self, shape, activation="softmax"):
        # input layer into keras model
        input_tensor_cr = Input(
            shape=shape, dtype="float32", name=self.name + "_input_cr"
        )
        input_tensor_adv = Input(
            shape=shape, dtype="float32", name=self.name + "_input_adv"
        )

        # setup up Dense layer
        output_tensor = Dense(3)
        output_tensor_cr = output_tensor(input_tensor_cr)
        output_tensor_adv = output_tensor(input_tensor_adv)
        output_tensor = Dense(3, activation=activation, name=self.name + "_output")
        output_tensor_cr = output_tensor(output_tensor_cr)
        output_tensor_adv = output_tensor(output_tensor_adv)

        return input_tensor_cr, output_tensor_cr, input_tensor_adv, output_tensor_adv
