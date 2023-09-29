from pathlib import Path

import pandas as pd
from cal_ratio_trainer.config import (
    ConvertDiVertAnalysisConfig,
    DiVertAnalysisInputFile,
    DiVertFileType,
    load_config,
)
from cal_ratio_trainer.convert.convert_divert import convert_divert


def test_empty_root_files(caplog, tmp_path):
    "run translation, make sure no crash occurs when a empty file is looked at"

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/empty_divert_analysis_file.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
            )
        ],
        output_path=tmp_path,
        signal_branches=["one", "two"],
        bib_branches=["three", "four"],
        qcd_branches=["five", "six"],
        llp_mH=125,
        llp_mS=100,
    )

    convert_divert(config)

    # Make sure the log message contains something about zero events.
    assert "0 events" in caplog.text


def test_missing_root_files(caplog, tmp_path):
    """run translation, make sure no crash occurs when a file with missing columns
    is looked at"""

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/short_divert_analysis_file.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
            )
        ],
        output_path=tmp_path,
        signal_branches=["one"],
        bib_branches=["one"],
        qcd_branches=["one"],
        llp_mH=125,
        llp_mS=100,
    )

    convert_divert(config)

    # Make sure the log message contains something about zero events.
    assert "one" in caplog.text
    assert "does not contain the required branches" in caplog.text


def test_no_redo_existing_file(caplog, tmp_path):
    "Second run, and nothing should happen"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/short_divert_analysis_file.root"),
                data_type=DiVertFileType.qcd,
                output_dir=None,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
        llp_mH=125,
        llp_mS=100,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "short_divert_analysis_file.pkl"
    assert output_file.exists()

    # Next, remove it and replace it with a zero length file, re-run,
    # and make sure it is still there.
    output_file.unlink()
    output_file.touch()

    convert_divert(config)

    assert output_file.exists()
    assert output_file.stat().st_size == 0


def test_bib_file(tmp_path, caplog):
    "Second run, and nothing should happen"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/bib.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
        llp_mH=0,
        llp_mS=0,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "bib.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) == 1


def test_sig_file(tmp_path, caplog):
    "Make sure a signal file runs correctly"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311424_600_275.root"),
                data_type=DiVertFileType.sig,
                output_dir=None,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
        llp_mH=0,
        llp_mS=0,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) == 76
