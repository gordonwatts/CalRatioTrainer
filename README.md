# CalRatioTrainer

Train the CalRatio 2019 RNN

## Introduction

This is far from production!

## Usage

This isn't meant to be an exploratory thing as much as "easy-to-run".

* `cr_trainer train --help` to see all the command line options.
* `cr_trainer train` will run the default (test) training. The test training has a reduced size signal file. However, everything is large enough to stress out the system. Good for running tests locally on your CPU.

Some quick notes:

* The first time you run this, it will copy down data files and cache them locally. You can change the cache location or file location using the configuration file.
* The output directory contains a complete set of the options that were used in the run, so it is easy to see exactly how a run was configured.
* test samples can run on a 16GB V100 if you do mini-match splitting of 15.

### Running Parameters

This is always an issue of trying to keep the number of min batches small to improve performance and now overflow your memory. Recorded below are a few running configurations:

* Laptop, i7, 32 GB, Windows, running on the test data: '--num_splits 2`.
* Chicago AF, V100 (16GB), 4 CPU with 32 GB, running on the test data: `--num_splits 10`.
* Chicago AF, V100 (16 GV), 4 CPU with 32 GB, running on the full Run 2 data: `--num_splits 230`. Running a full 100 epochs takes 2 hours.

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
