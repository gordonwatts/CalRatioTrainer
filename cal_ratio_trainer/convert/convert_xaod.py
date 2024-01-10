from typing import List
from cal_ratio_trainer.config import ConvertxAODConfig
import logging
import subprocess


def execute_commands(commands: List[str]) -> str:
    """Will execute a list of commands in the wsl2 instance.

    Args:
        commands (List[str]): List of commands to execute.
    """

    full_command = "; ".join(commands)
    output = []

    try:
        process = subprocess.Popen(
            [
                "wsl.exe",
                "-d",
                "atlas_centos7",
                "-e",
                "/bin/bash",
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


def do_checkout(default_directory: str):
    """Will check out the HEAD version of the DiVertAnalysis repo
    from GitHub, using the wsl2 `atlas_centos7` instance.
    """
    if dir_exists(default_directory):
        logging.debug("Directory already exists, only doing git update.")
        commands = [
            "cd ~/cr_trainer_DiVertAnalysis/src",
            "git pull",
            "git submodule update --init --recursive",
        ]
    else:
        commands = [
            "cd ~",
            "mkdir -p cr_trainer_DiVertAnalysis",
            "cd cr_trainer_DiVertAnalysis",
            "mkdir build run",
            "git clone --recursive ssh://git@gitlab.cern.ch:7999/atlas-phys-exotics-llp-mscrid/fullrun2analysis/DiVertAnalysisR21.git src",
        ]

    execute_commands(commands)


def delete_directory(dir: str):
    """Remove the directory and all its contents.

    Args:
        dir (str): linux path to the directory
    """
    commands = [f"rm -rf {dir}"]
    execute_commands(commands)


def convert_xaod(config: ConvertxAODConfig):
    """Will use the HEAD version of the DiVertAnalyusis repo to build
    and run the DiVertAnalysis executable. This will use `wsl2` to do
    the running (though one could easily configure this to be something
    else on Linux or a Mac!).

    Args:
        config (ConvertxAODConfig): Arguments for the conversion.
    """

    default_directory = config.working_directory

    # Do clean
    if config.clean:
        logging.info(f"Deleting directory {default_directory}")
        delete_directory(default_directory)

    # Do check out
    logging.info(f"Checking out DiVertAnalysis git package to {default_directory}")
    do_checkout(default_directory)

    # Do build

    # Do run
