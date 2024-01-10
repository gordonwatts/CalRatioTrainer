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
            "git clone --recursive ssh://git@gitlab.cern.ch:7999/atlas-phys-exotics-llp-mscrid/fullrun2analysis/DiVertAnalysisR21.git src",
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
        "export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase",
        "alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'",
    ]

    if already_setup:
        raise NotImplementedError()
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
    did_checkout = do_checkout(default_directory)

    # Do build
    do_build(default_directory, already_setup=not did_checkout)

    # Do run
