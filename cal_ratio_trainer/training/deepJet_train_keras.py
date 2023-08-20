from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Tuple
from cal_ratio_trainer.config import TrainingConfig
import tensorflow as tf
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput

from cal_ratio_trainer.training.utils import (
    create_directories,
    load_dataset,
    match_adversary_weights,
)


def train_llp(
    training_params: TrainingConfig,
    model_to_do: str,
    constit_input: ModelInput,
    track_input: ModelInput,
    MSeg_input: ModelInput,
    jet_input: JetInput,
    constit_input_adversary: ModelInput,
    track_input_adversary: ModelInput,
    MSeg_input_adversary: ModelInput,
    jet_input_adversary: JetInput,
    plt_model: bool = False,
) -> Tuple[float, str]:
    """
    Takes in arguments to change architecture of network, does training, then runs
        evaluate_training

    :param training_params: Class of many input parameters to training
    :param model_to_do: Name of the model
    :param useGPU2: True to use GPU2
    :param constit_input: ModelInput object for constituents
    :param track_input: ModelInput object for tracks
    :param MSeg_input: ModelInput object for muon segments
    :param jet_input: ModelInput object for jets
    :param plt_model: True to save model architecture to disk
    :param skip_training: Skip training and evaluate based on directory of network given
    """
    logging.info("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

    # Fail early if we aren't going to be able to find what we need.
    assert training_params.main_file is not None
    assert (
        training_params.main_file.exists()
    ), f"Main training file ({training_params.main_file}) does not exist"

    # Setup directories for output.
    logging.debug("Setting up directories...")
    dir_name = create_directories(
        model_to_do, os.path.split(os.path.splitext(training_params.main_file)[0])[1]
    )
    logging.debug(f"Main directory for output: {dir_name}")

    # Write a file with some details of architecture, will append final stats at end
    # of
    # training
    logging.debug("Writing to file training details...")
    training_details_file = Path("plots/") / dir_name / "training_details.txt"
    with training_details_file.open("wt+") as f_out:
        print(f"Starting Training {datetime.now()}", file=f_out)
        print(f"ModeL: {model_to_do}", file=f_out)
        print(str(training_params), file=f_out)

    # Load dataset
    logging.debug(f"Loading control region data file {training_params.cr_file}...")
    df_adversary = load_dataset(training_params.cr_file)
    assert training_params.frac_list is not None
    df_adversary = match_adversary_weights(df_adversary)
    # TODO: random seed isn't passed in here to `sample`, so it will
    # be different each run.
    df_adversary = df_adversary.sample(frac=training_params.frac_list)

    logging.debug(f"Loading up dataset {training_params.main_file}...")
    df = load_dataset(training_params.main_file)
    df = df.sample(frac=training_params.frac_list)

    # # Extract labels
    # (
    #     X,
    #     X_adversary,
    #     Y,
    #     Y_adversary,
    #     mcWeights,
    #     mcWeights_adversary,
    #     sig_weight,
    #     weights,
    #     weights_adversary,
    #     Z,
    #     Z_adversary,
    # ) = prepare_training_datasets(df, df_adversary)

    # """
    # debug check to see if columns are the same
    # (alex) I think this is an unnecessary check
    # """
    # # X_cols = list(X.columns)
    # # X_adv_cols = list(X_adversary.columns)
    # # bool_list = list(map(lambda x, y: x == y, X_cols, X_adv_cols))
    # # #X_bool_list = X[bool_list]
    # # #X_adv_bool_list = X_adversary[bool_list]
    # # if X_cols == X_adv_cols:
    # #     print("X, X_adv columns are the same, good!!")
    # # else:
    # #     print("Warning: Columns in X and X_adv not same!\nCould just be a name "
    # #     "mis-match, but please check!")
    # #     #exit()

    # # Save memory
    # del df
    # del df_adversary
    # gc.collect()

    # # Handle case if no KFold
    # if kfold is None:
    #     # Split data into train/test datasets
    #     random_state = 1
    #     (
    #         X_train,
    #         X_test,
    #         y_train,
    #         y_test,
    #         weights_train,
    #         weights_test,
    #         mcWeights_train,
    #         mcWeights_test,
    #         Z_train,
    #         Z_test,
    #     ) = train_test_split(
    #         X,
    #         Y,
    #         weights,
    #         mcWeights,
    #         Z,
    #         test_size=0.1,
    #         random_state=random_state,
    #         shuffle=False,
    #     )
    #     (
    #         X_train_adversary,
    #         X_test_adversary,
    #         y_train_adversary,
    #         y_test_adversary,
    #         weights_train_adversary,
    #         weights_test_adversary,
    #         mcWeights_train_adversary,
    #         mcWeights_test_adversary,
    #         Z_train_adversary,
    #         Z_test_adversary,
    #     ) = train_test_split(
    #         X_adversary,
    #         Y_adversary,
    #         weights_adversary,
    #         mcWeights_adversary,
    #         Z_adversary,
    #         test_size=0.1,
    #         random_state=random_state,
    #         shuffle=False,
    #     )

    #     # Delete variables to save memory
    #     del X
    #     del X_adversary
    #     gc.collect()
    #     del Y
    #     del Y_adversary
    #     gc.collect()
    #     del Z
    #     del Z_adversary
    #     gc.collect()
    #     eval_object = evaluationObject()
    #     eval_object.fillObject_params(
    #         training_params.frac_list,
    #         training_params.batch_size,
    #         training_params.reg_values,
    #         training_params.dropout_array,
    #         training_params.epochs,
    #         training_params.lr_values,
    #         training_params.hidden_layer_fraction,
    #         int(constit_input.filters_cnn[3] / 4),
    #         sig_weight,
    #         int(constit_input.nodes_lstm / 40),
    #     )

    #     # Call method that prepares data, builds model architecture, trains model,
    #     # and evaluates model
    #     roc_auc = build_train_evaluate_model(
    #         constit_input,
    #         track_input,
    #         MSeg_input,
    #         jet_input,
    #         X_train,
    #         X_test,
    #         y_train,
    #         y_test,
    #         mcWeights_train,
    #         mcWeights_test,
    #         weights_train,
    #         weights_test,
    #         Z_test,
    #         Z_train,
    #         constit_input_adversary,
    #         track_input_adversary,
    #         MSeg_input_adversary,
    #         jet_input_adversary,
    #         X_train_adversary,
    #         X_test_adversary,
    #         y_train_adversary,
    #         y_test_adversary,
    #         mcWeights_train_adversary,
    #         mcWeights_test_adversary,
    #         weights_train_adversary,
    #         weights_test_adversary,
    #         Z_test_adversary,
    #         Z_train_adversary,
    #         plt_model,
    #         dir_name,
    #         eval_object,
    #         useGPU2,
    #         skipTraining,
    #         training_params,
    #     )

    #     f.close()
    #     return roc_auc, dir_name

    # else:
    #     # initialize lists to store metrics
    #     roc_scores, acc_scores = list(), list()
    #     # initialize counter for current fold iteration
    #     n_folds = 0
    #     # do KFold Cross Validation
    #     eval_object_list = []
    #     for train_ix, test_ix, train_adv_ix, test_adv_ix in (
    #         kfold.split(X, Y),
    #         kfold.split(X_adversary, Y_adversary),
    #     ):
    #         eval_object = evaluationObject()
    #         eval_object.fillObject_params(
    #             training_params.frac_list,
    #             training_params.batch_size,
    #             training_params.reg_values,
    #             training_params.dropout_array,
    #             training_params.epochs,
    #             training_params.lr_values,
    #             training_params.hidden_layer_fraction,
    #             int(constit_input.filters_cnn[3] / 4),
    #             sig_weight,
    #             int(constit_input.nodes_lstm / 40),
    #         )
    #         n_folds += 1
    #         print("\nDoing KFold iteration # %.0f...\n" % n_folds)
    #         # select samples
    #         X_test, y_test, weights_test, mcWeights_test, Z_test = (
    #             X.iloc[test_ix],
    #             Y.iloc[test_ix],
    #             weights.iloc[test_ix],
    #             mcWeights.iloc[test_ix],
    #             Z.iloc[test_ix],
    #         )
    #         X_train, y_train, weights_train, mcWeights_train, Z_train = (
    #             X.iloc[train_ix],
    #             Y.iloc[train_ix],
    #             weights.iloc[train_ix],
    #             mcWeights.iloc[train_ix],
    #             Z.iloc[train_ix],
    #         )

    #         (
    #             X_test_adversary,
    #             y_test_adversary,
    #             weights_test_adversary,
    #             mcWeights_test_adversary,
    #             Z_test_adversary,
    #         ) = (
    #             X_adversary.iloc[test_adv_ix],
    #             Y_adversary.iloc[test_adv_ix],
    #             weights_adversary.iloc[test_adv_ix],
    #             mcWeights_adversary.iloc[test_adv_ix],
    #             Z_adversary.iloc[test_adv_ix],
    #         )
    #         (
    #             X_train_adversary,
    #             y_train_adversary,
    #             weights_train_adversary,
    #             mcWeights_train_adversary,
    #             Z_train_adversary,
    #         ) = (
    #             X_adversary.iloc[train_adv_ix],
    #             Y_adversary.iloc[train_adv_ix],
    #             weights_adversary.iloc[train_adv_ix],
    #             mcWeights_adversary.iloc[train_adv_ix],
    #             Z_adversary.iloc[train_adv_ix],
    #         )

    #         # Call method that prepares data, builds model architecture, trains model,
    #         # and evaluates model
    #         roc_auc = build_train_evaluate_model(
    #             constit_input,
    #             track_input,
    #             MSeg_input,
    #             jet_input,
    #             X_train,
    #             X_test,
    #             y_train,
    #             y_test,
    #             mcWeights_train,
    #             mcWeights_test,
    #             weights_train,
    #             weights_test,
    #             Z_test,
    #             Z_train,
    #             constit_input_adversary,
    #             track_input_adversary,
    #             MSeg_input_adversary,
    #             jet_input_adversary,
    #             X_train_adversary,
    #             X_test_adversary,
    #             y_train_adversary,
    #             y_test_adversary,
    #             mcWeights_train_adversary,
    #             mcWeights_test_adversary,
    #             weights_train_adversary,
    #             weights_test_adversary,
    #             plt_model,
    #             dir_name,
    #             eval_object,
    #             useGPU2,
    #             skipTraining,
    #             training_params,
    #             kfold,
    #             n_folds,
    #         )

    #         roc_scores.append(roc_auc)
    #         eval_object_list.append(eval_object)
    #         gc.collect()
    #     evaluate_objectList(eval_object_list, f)
    #     f.close()
    #     return roc_scores, dir_name

    return 10, dir_name
