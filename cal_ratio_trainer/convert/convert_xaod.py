import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import shutil

from cal_ratio_trainer.config import ConvertxAODConfig


def execute_commands(commands: List[str]) -> str:
    """Will execute a list of commands in the wsl2 instance.

    Args:
        commands (List[str]): List of commands to execute.
    """

    full_command = "; ".join(commands)
    output = []

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

    This code will need to be fixed depending on starting and ending
    places!

    Args:
        wsl_path (str): Path to the file in the WSL instance.
        local_path (Path): Path to the file locally.
    """
    source_file = Path(wsl_path)
    commands = [f"cp {wsl_path} /mnt/wsl/{source_file.name}"]
    execute_commands(commands)

    temp_file = Path(f"/mnt/wsl/{source_file.name}")
    shutil.move(temp_file, local_path)


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

        # Do build
        do_build(default_directory, already_setup=not did_checkout)

    # Do run
    do_run(
        default_directory,
        files=config.input_files,
        n_events=config.nevents,
        output_file=config.output_path,
    )
