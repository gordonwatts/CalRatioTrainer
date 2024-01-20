import logging
from math import log
import subprocess
from pathlib import Path
from typing import List, Optional
import shutil

from cal_ratio_trainer.config import ConvertTrainingConfig, ConvertxAODConfig
from cal_ratio_trainer.convert.convert_json import convert_file


def execute_commands(commands: List[str]) -> str:
    """Will execute a list of commands in the wsl2 instance.

    Args:
        commands (List[str]): List of commands to execute.
    """

    full_command = "; ".join(commands)
    output = []
    logging.debug(f"Executing command: {full_command}")

    # TODO: #204 Make xaod conversion work on regular linux and windows, etc.

    try:
        process = subprocess.Popen(
            [
                "wsl.exe",
                "-d",
                "atlas_centos7",
                "-e",
                "/bin/bash",
                "-i",
                "-c",
                f"{full_command}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            universal_newlines=True,
            bufsize=1,  # Line-buffered output
        )

        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            output.append(line.strip())
            logging.debug(line.strip())

        process.wait()
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error executing command: {full_command}")
        logging.warning(e.output.strip())
        raise

    if process.returncode != 0:
        logging.warning(f"Error executing command: {full_command}")
        logging.warning("\n".join(output))
        raise Exception(f"Error executing command: {full_command}")

    return "\n".join(output)


def dir_exists(directory: str) -> bool:
    """Will check if a directory exists in the wsl2 instance.

    Args:
        directory (str): Directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    r = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "atlas_centos7",
            "-e",
            "/bin/bash",
            "-c",
            f"test -d {directory}",
        ],
        check=False,
    )
    return r.returncode == 0


def text_exists_in_file(file: str, text: str) -> bool:
    """Will check if a text exists in a file in the wsl2 instance.

    Args:
        file (str): File to check.
        text (str): Text to check for.

    Returns:
        bool: True if the text exists, False otherwise.
    """
    r = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "atlas_centos7",
            "-e",
            "/bin/bash",
            "-c",
            f"grep -q {text} {file}",
        ],
        check=False,
    )
    return r.returncode == 0


def do_checkout(default_directory: str) -> bool:
    """Will check out the HEAD version of the DiVertAnalysis repo
    from GitHub, using the wsl2 `atlas_centos7` instance.
    """
    result = False
    if dir_exists(default_directory):
        logging.debug("Directory already exists, only doing git update.")
        commands = [
            "cd ~/cr_trainer_DiVertAnalysis/src",
            "git pull",
            "git submodule update --init --recursive",
        ]
        result = False
    else:
        logging.debug("Directory does not exist, doing full git clone.")
        commands = [
            "cd ~",
            "mkdir -p cr_trainer_DiVertAnalysis",
            "cd cr_trainer_DiVertAnalysis",
            "mkdir build run",
            "git clone --recursive ssh://git@gitlab.cern.ch:7999"
            "/atlas-phys-exotics-llp-mscrid/fullrun2analysis/DiVertAnalysisR21.git src",
            "cd src/FactoryTools",
            "source util/dependencyHacks.sh",
        ]
        result = True

    execute_commands(commands)
    return result


def do_build(dir: str, already_setup: bool):
    """Execute the build commands - and how we setup depends if we have already
    run.

    Args:
        dir (str): Working directory for the build.
        already_setup (bool): True if this was already configured and compiled.
    """
    # Make sure we are setup.

    config_atlas_commands = [
        "setupATLAS -2",
    ]

    if already_setup:
        setup_commands = []
    else:
        setup_commands = [
            f"cd {dir}/src/FactoryTools",
            "source util/setup.sh",
        ]

    # Now do the build.
    commands = [
        f"cd {dir}/src",
        "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$PWD/DiVertAnalysis/externals/include/",
        "cd ../build/",
    ]

    if already_setup:
        commands += [
            "asetup --restore",
        ]

    commands += [
        "cmake ../src/",
        "cmake --build .",
    ]

    execute_commands(config_atlas_commands + setup_commands + commands)


def delete_directory(dir: str):
    """Remove the directory and all its contents.

    Args:
        dir (str): linux path to the directory
    """
    logging.debug(f"Deleting directory {dir}")
    commands = [f"rm -rf {dir}"]
    execute_commands(commands)


def convert_to_wsl_path(local_path: Path) -> str:
    """Convert a windows to a WSL path.

    Args:
        path (Path): Path to convert.

    Returns:
        str: WSL path.
    """
    return str(local_path)
    # drive, path = os.path.splitdrive(str(local_path))
    # path = path.replace("\\", "/")
    # return f"/mnt/{drive.lower()[0]}{path}"


def copy_file_locally(wsl_path: str, local_path: Path):
    """Copy a file from the WSL instance to another WSL instance.

    NOTE: This code will need to be fixed depending on starting and ending
    places! For now assume wsl2 as the starting target, and this
    instance as the final one.

    Args:
        wsl_path (str): Path to the file in the WSL instance.
        local_path (Path): Path to the file locally.
    """
    logging.debug(f"Copying wsl2:{wsl_path} to {local_path}")
    source_file = Path(wsl_path)
    commands = [f"cp {wsl_path} /mnt/wsl/{source_file.name}"]
    execute_commands(commands)

    temp_file = Path(f"/mnt/wsl/{source_file.name}")
    shutil.move(temp_file, local_path)


def copy_local_file(local_path: Path, wsl_path: str):
    """Copy a file from the local machine to the WSL instance.

    This code will need to be fixed depending on starting and ending
    places! For now assume this instance as the starting place,
    and wsl2 as the remote one.

    Args:
        local_path (Path): Path to the file locally.
        wsl_path (str): Path to the file in the WSL instance.
    """
    logging.debug(f"Copying {local_path} to wsl2:{wsl_path}")
    # Put it in the common wsl location.
    temp_file = Path(f"/mnt/wsl/{Path(local_path).name}")
    shutil.copy(local_path, temp_file)

    commands = [f"mv {temp_file} {wsl_path}"]
    execute_commands(commands)


def do_run(
    directory: str, files: List[Path], n_events: Optional[int], output_file: Path
):
    """Run the DiVertAnalysis executable.

    Args:
        directory (str): The directory containing the DiVertAnalysis executable.
        files (List[str]): List of input files.
    """
    # Build the command line:
    divert_command = (
        f"produce_ntuple.py -c mc16e -t signal -a cr --nopileup"
        f" --inputDS {convert_to_wsl_path(files[0])}"
    )

    if n_events is not None:
        divert_command += f" --nevents {n_events}"

    setup_commands = [
        "setupATLAS -2",
        f"cd {directory}/src",
        "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$PWD/DiVertAnalysis/externals/include/",
        "cd ../build/",
        "asetup --restore",
        "source */setup.sh",
    ]

    # Next, setup and run everything:
    commands = [f"cd {directory}/run", "rm -rf submit_dir", divert_command]

    execute_commands(setup_commands + commands)

    # TODO: capture the output from the `execute_commands`, and if the copy below fails,
    # we should dump the output so the user can see how the thing ran (or didn't run).
    # This is b.c. the above dose not throw an exception if the job fails b.c. it
    # doesn't return a bad shell status.

    # Final thing is to copy the output file back locally.
    copy_file_locally(
        f"{directory}/run/submit_dir/data-trees/{files[0].name}.root", output_file
    )


def add_training(default_directory: str, name: str, path: str):
    """Add a training to the DiVertAnalysis code.

    Args:
        default_directory (Path): Path to the DiVertAnalysis code.
        name (str): Name of the training.
        path (str): Path to the training.
    """
    # If the file has already been added, do not add again!
    header_file = (
        f"{default_directory}/src/DiVertAnalysis/DiVertAnalysis/"
        "RegionVarCalculator_calRatio.h"
    )
    if text_exists_in_file(header_file, name):
        logging.info(f"Training {name} already exists in {header_file}")
    else:
        logging.info(f"Adding training {name} to {header_file}")

        new_line = f'    {{"{name}", initialize_fdeep_model("{path}")}},'

        sed_command = (
            f"sed -i '/keras_model_highMass_v3Adv_apr28/a {new_line}' {header_file}"
        )
        execute_commands([sed_command])


def convert_xaod(config: ConvertxAODConfig):
    """Will use the HEAD version of the DiVertAnalyusis repo to build
    and run the DiVertAnalysis executable. This will use `wsl2` to do
    the running (though one could easily configure this to be something
    else on Linux or a Mac!).

    Args:
        config (ConvertxAODConfig): Arguments for the conversion.
    """

    default_directory = config.working_directory
    assert config.input_files is not None, "You must list input files to convert!"
    assert config.output_path is not None, "You must specify an output path!"
    assert (
        not config.output_path.exists()
    ), f"Output path {config.output_path} already exists. Please delete before running"
    assert not (
        config.skip_build and config.clean
    ), "Cannot skip build and ask for a clean start!"

    # Do clean
    if not config.skip_build:
        if config.clean:
            logging.info(f"Deleting directory {default_directory}")
            delete_directory(default_directory)

        # Do check out
        logging.info(f"Checking out DiVertAnalysis git package to {default_directory}")
        did_checkout = do_checkout(default_directory)

        # Next, lets see if we need to add a NN to the list of NN's that are there.
        assert config.add_training is not None
        for nn in config.add_training:
            # we need to convert the training file to json that
            # can be used by the DiVertAnalysis Code.
            output_file = f"/tmp/nn_config_{nn.name}_{nn.run}_{nn.epoch}.json"
            convert_config = ConvertTrainingConfig(
                run_to_convert=nn, output_json=Path(output_file)
            )
            print("Doing conversion...")
            convert_file(convert_config)

            # Copy it from this machine to the WSL instance.
            copy_local_file(
                Path(output_file),
                f"{default_directory}/src/DiVertAnalysis/data/"
                "nn_config_{nn.name}_{nn.run}_{nn.epoch}.json",
            )

            # Finally, add the config line to the C++ header file so it gets used.
            add_training(
                default_directory,
                f"{nn.name}_{nn.run}_{nn.epoch}",
                f"/DiVertAnalysis/nn_config_{nn.name}_{nn.run}_{nn.epoch}.json",
            )
        exit(1)

        # Do build
        do_build(default_directory, already_setup=not did_checkout)

    # Do run
    do_run(
        default_directory,
        files=config.input_files,
        n_events=config.nevents,
        output_file=config.output_path,
    )
