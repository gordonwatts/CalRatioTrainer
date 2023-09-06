# CalRatioTrainer

Train the CalRatio 2019 RNN

## Introduction

This is far from production!

## Usage

This isn't meant to be an exploratory thing as much as "easy-to-run".

* `cr_trainer train --help` to see all the command line options.
* `cr_trainer train` will run the default (test) training. The test training has a reduced size signal file. However, everything is large enough to stress out the system. Good for running tests locally on your CPU. Results are stored under `training_results`, which is created in your current directory.

Some quick notes:

* The first time you run this, it will copy down data files and cache them locally. You can change the cache location or file location using the configuration file.
* The output directory contains a complete set of the options that were used in the run, so it is easy to see exactly how a run was configured.
* test samples can run on a 16GB V100 if you do mini-match splitting of 15.

### Running Parameters

This is always an issue of trying to keep the number of min batches small to improve performance and now overflow your memory. Recorded below are a few running configurations:

* Laptop, i7, 32 GB, Windows, running on the test data: '--num_splits 2`.
* Chicago AF, V100 (16 GB), 4 CPU with 32 GB, running on the test data: `--num_splits 10`.
* Chicago AF, V100 (16 GV), 4 CPU with 32 GB, running on the full Run 2 data: `--num_splits 230`. Running a full 100 epochs takes 2 hours.
* Chicago AF, A100 (40 GB), 4 CPU with 32 GB, running on the full Run 2 data: `--num_splits 7`. Not clear this is well behaved from a training POV, however.

### Data

The following training datasets are used:

| Dataset Name | Source | Description |
| --- | --- | --- |
| X | `main_training_file` | The main Training File |

The following dataflow diagram attempts to follow the flow of training and control input data through the algorithm.

```mermaid
graph LR;
    main_training_file-->X;
    main_training_file-->Z;
    main_training_file-->weights;
    weights-->|reweighed to signal| mcWeights;
    weights-->|90% split| weights_train;
    weights-->|10% split|weights_test;
    mcWeights-->|90% split|mcWeights_train;
    mcWeights-->|10% split|mcWeights_test;
    cr_training_file-->X_adversary;
    cr_training_file-->Z_adversary;
    cr_training_file-->weights_adversary;
    cr_training_file-->|reweight to signal|mcWeights_adversary;
    weights_adversary-->|90% split|weights_train_adversary;
    weights_adversary-->|10% split|weights_test_adversary;
    mcWeights_adversary-->|90% split|mcWeights_train_adversary;
    mcWeights_adversary-->|10% split|mcWeights_test_adversary;
    X-->|90% split|X_train;
    X-->|10% split|X_test;
    Z-->|90% split|Z_train;
    Z-->|10% split|Z_test;
    X_adversary-->|90% split|X_train_adversary;
    X_adversary-->|10% split|X_test_adversary;
    Z_adversary-->|90% split|Z_train_adversary;
    Z_adversary-->|10% split|Z_test_adversary;
    X_test-->|50% split|X_test2[X_test];
    X_test-->|50% split|X_val;
    weights_test-->|50% split|weights_test2[weights_test];
    weights_test-->|50% split|weights_val;
    mcWeights_test-->|50% split|mcWeights_test2[mcWeights_test];
    mcWeights_test-->|50% split|mcWeights_val;
    Z_test-->|50% split|Z_test2[Z_test];
    Z_test-->|50% split|Z_val;
    X_test_adversary-->|50% split|X_test_adversary2[x_test_adversary];
    X_test_adversary-->|50% split|X_val_adversary[x_val_adversary];
    weights_test_adversary-->|50% split|weights_test_adversary2[weights_test_adversary];
    weights_test_adversary-->|50% split|weights_val_adversary;
    mcWeights_test_adversary-->|50% split|mcWeights_test_adversary2[mcWeights_test_adversary];
    mcWeights_test_adversary-->|50% split|mcWeights_val_adversary;
    X_train-->X_train2[X_train hi/lo mass, pad];
    Z_train-->X_train2;
    Z_train-->Z_train2[Z_train hi/lo mass, pad];
    weights_train-->weights_train2[weights_train hi/lo mass, pad];
    Z_train-->weights_train2
    mcWeights_train-->mcWeights_train2[mcWeights_train hi/lo mass, pad];
    Z_train-->mcWeights_train;
    Z_train0-->Z_train2[Z_train hi/lo mass, pad];

    X_val_adversary-->|training vars|x_to_validate_adv;

    mcWeights_val_adversary-->small_mcWeights_val_adversary;
    weights_val_adversary-->small_weights_val_adversary;
    x_to_validate_adv-->small_x_val_adversary;
    weights_train_adversary_s-->small_weights_train_adversary;
    x_to_adversary-->small_x_to_adversary;

    x_to_adversary-->|pad| x_to_adversary2[x_to_adversary];
    x_to_train-->|pad value| x_to_adversary2;

    x_to_validate_adv-->|pad| x_to_validate_adv2[x_to_validate_adv];
    x_to_validate-->|pad value| x_to_validate_adv;

    weights_train_adversary-->|pad| weights_train_adversary_s[weights_train_adversary];
    weights_to_train-->|pad value| weights_train_adversary2;

    weights_val_adversary-->|pad|weights_val_adversary_values2[weights_val_adversary_values];
    weights_to_validate-->weights_val_adversary_values2

    x_to_train-->|split by mini-batch|x_to_train_split;
    x_to_adversary2-->|split by mini-batch|x_to_adversary_split;

    weights_to_train-->|split by mini-match| weights_to_train_0;
    weights_train_adversary2-->|split by mini-batch| small_x_to_adversary_split;

    small_weights_train_adversary-->|split by mini-batch| small_weights_train_adversary_s;

    x_to_validate-->|split by mini-batch| x_to_validate_split;
    x_to_validate_adv2-->|split by mini-batch| x_to_validate_adv_split;

    weights_to_validate-->|split by mini-batch| weights_to_validate_0;
    weights_val_adversary-->|split by mini-batch| weights_val_adversary_split;

    small_x_to_adversary_split-->|mini-batch| discriminator_model[discriminator_model training];
    small_y_to_train_adversary_0-->|mini-batch| discriminator_model;
    small_weights_train_adversary-->|mini-batch| discriminator_model;
    discriminator_model-->last_disc_loss;
    discriminator_model-->last_disc_bin_acc;

    x_to_train_split-->|mini-batch| train_inputs;
    x_to_adversary_split-->|mini-batch| train_inputs;

    y_to_train_0-->|mini-batch| train_outputs;
    y_to_train_adversary_squeeze-->|mini-batch| train_outputs

    weights_to_train_0-->|mini-batch| train_weights;
    weights_train_adversary_s-->|mini-batch| train_weights;

    train_inputs-->original_model[original_model training];
    train_outputs-->original_model;
    train_weights-->original_model;
    original_model-->last_loss;
    original_model-->last_main_output_loss;
    original_model-->last_adversary_loss;
    original_model-->last_main_cat_acc;
    original_model-->last_adv_bin_acc;

    x_to_validate_split-->|mini-batch 0|original_model2[evaluate original_model];
    x_to_validate_adv_split-->|mini-batch 0|original_model2;
    weights_to_validate_0-->|mini-batch 0|original_model2;
    weights_val_adversary_split-->|mini-batch 0|original_model2;
    original_model2-->val_last_loss;
    original_model2-->val_last_main_output_loss;
    original_model2-->val_last_adversary_loss;
    original_model2-->val_last_main_cat_acc;
    original_model2-->val_last_adv_bin_acc;

    small_x_val_adversary-->discriminator_model[evaluate discriminator_model];
    small_weights_val_adversary-->discriminator_model[evaluate discriminator_model];

    discriminator_model-->|Adversary Loss|file_test_adv_loss[test_adv_loss.png];
    discriminator_model-->|Adversary Accuracy|file_adv_acc[adv_acc.png];

    style file_test_adv_loss fill:#f00,stroke:#333,stroke-width:4px;
    style file_adv_acc fill:#f00,stroke:#333,stroke-width:4px;

    small_x_val_adversary-->|signal| file_epoch_val_adversary_sig_prediction[nnn_val_adversary_sig_predictions];
    small_x_val_adversary-->|QCD| file_epoch_val_adversary_qcd_prediction[nnn_val_adversary_qcd_predictions];
    small_x_val_adversary-->|BIB| file_epoch_val_adversary_bib_prediction[nnn_val_adversary_bib_predictions];
    small_mcWeights_val_adversary-->|signal| file_epoch_val_adversary_sig_prediction;
    small_mcWeights_val_adversary-->|QCD| file_epoch_val_adversary_qcd_prediction;
    small_mcWeights_val_adversary-->|BIB| file_epoch_val_adversary_bib_prediction;

    style file_epoch_val_adversary_sig_prediction fill:#f00,stroke:#333,stroke-width:4px;
    style file_epoch_val_adversary_qcd_prediction fill:#f00,stroke:#333,stroke-width:4px;
    style file_epoch_val_adversary_bib_prediction fill:#f00,stroke:#333,stroke-width:4px;

    small_x_val_adversary-->ks_test;
    small_mcWeights_val_adversary-->ks_test;

    ks_test-->|qcd|ks_qcd;
    ks_test-->|BIB|ks_bib;
    ks_test-->|signal|ks_sig;

    ks_qcd-->file_ks_qcd[ks_qcd.png];
    ks_bib-->file_ks_bib[ks_bib.png];
    ks_sig-->file_ks_sig[ks_sig.png];
    style file_ks_qcd fill:#f00,stroke:#333,stroke-width:4px;
    style file_ks_bib fill:#f00,stroke:#333,stroke-width:4px;
    style file_ks_sig fill:#f00,stroke:#333,stroke-width:4px;

    X_test2-->final_model[evaluate final_model];

    final_model-->file_epoch_main_sig_prediction[nnn_main_sig_predictions];
    mcWeights_test2-->file_epoch_main_sig_prediction;
    final_model-->file_epoch_main_qcd_prediction[nnn_main_qcd_predictions];
    mcWeights_test2-->file_epoch_main_qcd_prediction;
    final_model-->file_epoch_main_bib_prediction[nnn_main_bib_predictions];
    mcWeights_test2-->file_epoch_main_bib_prediction;

    style file_epoch_main_sig_prediction fill:#f00,stroke:#333,stroke-width:4px;
    style file_epoch_main_qcd_prediction fill:#f00,stroke:#333,stroke-width:4px;
    style file_epoch_main_bib_prediction fill:#f00,stroke:#333,stroke-width:4px;

    X_test2-->|most recent epoch|final_model_2[evaluate final_model]
    final_model_2-->file_sig_predictions[sig, qcd, bib_predictions"", _linear, _half_linear];
    mcWeights_test-->file_sig_predictions;

    style file_sig_predictions fill:#f00,stroke:#333,stroke-width:4px;

    X_test_adversary2-->|most recent epoch|final_model_3[evaluate final_model]
    final_model_3-->file_adv_prediction[adv_sig, bkg, bib_prediction];
    weights_test_adversary2-->file_adv_prediction;

    style file_adv_prediction fill:#f00,stroke:#333,stroke-width:4px;

    last_adv_bin_acc-->file_main_adv_acc;
    val_last_adv_bin_acc-->file_main_adv_acc;
    style file_main_adv_acc fill:#f00,stroke:#333,stroke-width:4px;

    last_main_output_loss-->file_main_nn_loss;
    val_last_main_output_loss-->file_main_nn_loss;
    style file_main_nn_loss fill:#f00,stroke:#333,stroke-width:4px;

    last_main_cat_acc-->file_main_network_acc;
    val_last_main_cat_acc-->file_main_network_acc;
    style file_main_network_acc fill:#f00,stroke:#333,stroke-width:4px;

    X_test2-->file_roc_and_soverb[ROC and SoverB Plots];
    Z_test2-->file_roc_and_soverb;
    mcWeights_test2-->file_roc_and_soverb;
    style file_roc_and_soverb fill:#f00,stroke:#333,stroke-width:4px;
```

Notes:

* A `2` at the end of the same name, `X_test2` means that the variable `X_test` was replaced (and so on).
* There are three catagories for the main data, `0: Signal`, `1: MC Mulitjet`, and `2: BIB`.
* There are two catagories for the adversary data, `0:xxx` and `1:xxx`
* `Y` variables are not mentioned as they contain the "truth".
* `X` is all columns including jet info and clusterSS, and track, and muon segment.
* `Z` is the LLP truth information (for parameterize training?)
* `weights` are the raw weights that come from the file we read in. `mcWeights` is rescaled so QCD and Signal have the same weight.
* The `small` data variables are basically the unpadded/unextended data. They are often used to evaluate the discriminator.

### Plots

By default, as the training runs, a great deal of plots are produced. This list below is an attempt to understand those plots.

* The `keras` directory contains a copy of the model and check points of the training parameters. The training parameters aren't written for every epoch, only where the K-S test for BIB is below `0.3` (see below). The `checkpoint` files are written after every epoch and give you the most recently completed weights, good or bad.
* The output directory for the run contains lots of files that begin with an integer - these.

#### Per-Epoch Plots

15 plots are produced each epoch to make for easy tracking.

| file-name | Description |
| --- | --- |
| `<nnn>_main__(bib, qcd, sig)_predictions_linear` | Each plot shows one of the three outputs of the NN when run on xxx by the type of data. Excellent to see the performance: one expects the signal to be piled at the right, for example, for the signal output of the NN. The test data is used to generate these plots. |
| `<nnn>_val_adversary__(bib, qcd, sig)_predictions` | Same plots, but using the `small_val_adversary` dataset, which is half the dataset that was originally used for testing. This are on the adversary dataset, with only data and multijet MC (you'll note there is no BIB in these plots). Do not be fooled by the legend text |
| `<nnn>_val_adversary_(highPt, midPt, lowPt)_(bib, qcd, sig)_predictions` | Same as the `val_adversary` plots above, but split by $p_T$. Low is $p_T < 0.25$, mid is $0.25 < p_T < 0.50$, and high is $p_T > 0.5$. |
| `<nnn>_main_(bib, qcd, sig)_predictions` | The main network output distributions for each of the output variables (bib, qcd, and signal). In each plot, if the training is working well, you should see the bib pushed up against the right edge of the bib NN output, and same for QCD for the QCD NN output, etc. Good to check to see if the network is learning how to differentiate between signal and its two backgrounds. |

* $p_T$ is rescaled to xx. This means 0.25 is xx, and 0.50 is yy.

#### Final Plots

| file-name | Description |
| --- | --- |
| `main_nn_loss` | The loss from the main network on test data and training data. Can check by-eye for performance and (see warning) overtraining. Dumped from `original_lossf` and `val_original_lossf`. The validation dataset is the full dataset. WARNING (TODO): The main loss is only the last mini-batch and so will be statistically limited! |
| `ks_(bib, qcd, sig)` | The K-S test results per epoch. Calculated in the `do_checkpoint_prediction_histogram` method (called once per epoch). |
| `(qcd, bib, sig)_signal_predictions` | |
| `main_adv_acc` | The loss of the adversary network on the test and training datasets. |

## Installation

Installation instructions are generally tricky: this really needs to be trained on a GPU.

### WSL2

This is without using the GPU (so good for testing).

1. Open up an instance. If you are using the full Run 2 dataset, then you'll need the 40GB instance to be as efficient as possible.
1. Open a terminal window
1. `git clone https://github.com/gordonwatts/CalRatioTrainer.git` into whatever directory you want to run out of, in a new virtual environment.
1. `cd CalRatioTrainer`
1. `pip install -e .[wsl2]`
    * If you want to do development, etc., then do `pip install -e .[test,wsl2]`

This should work anywhere you are using a clean environment. It will install `TensorFlow`, for example. It is always a fight getting the right version of TF and the underlying GPU libraries to work together, so you may have to fiddle after the install depending on your setup. Feel free to submit PR's if you find something that might be interesting to others!

### Chicago Analysis Facility

The installation is expected to take place on a Jupyter instance where the proper TF libraries have already been installed.

1. Open up an instance. If you are using the full Run 2 dataset, then you'll need the 40GB instance to be as efficient as possible.
1. Open a terminal window
1. `git clone https://github.com/gordonwatts/CalRatioTrainer.git` into whatever directory you want to run out of.
1. `cd CalRatioTrainer`
1. `pip install -e .`
    * If you want to do development, etc., then do `pip install -e .[test]`

You should be ready to go!

## Acknowledgements

This is based on the work originally done by Felix in the CalRatio group in ATLAS. This RNN was published in xxx.
The running and design has been improved since then.

Changes from Felix's original code.

Cosmetic:

* Only code directly used to do the training, etc., was copied over.
* All code is formatted using `black` and `flake8` for readability.
* Sub-commands using `argparse` are used to control
* Use `pydantic` to steer the training, and allow for command line arguments to be used.
* Use the directory `training_results` to store all results. That directory contains the `model_name`,
  and under that the run number.
* It is possible to "continue" a training from a previous one. See help strings for `--continue-n` from the command help `cr_trainer train --help`.

Algorithmic:

* Do not recompile the model every or during all the adversary during mini-batches.
* Do not change the learning rate as a function of the epoch
* Removed cross-validation code

Typical training takes about 40 minutes on the full dataset, 100 epochs.
