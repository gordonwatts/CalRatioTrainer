from pathlib import Path
from cal_ratio_trainer.config import (
    ConvertDiVertAnalysisConfig,
    DiVertAnalysisInputFile,
    DiVertFileType,
)
from cal_ratio_trainer.convert.convert_divert import convert_divert


def test_empty_root_files(caplog):
    "run translation, make sure no crash occurs when a empty file is looked at"

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/empty_divert_analysis_file.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
            )
        ],
        output_path=Path("./out"),
        signal_branches=["one", "two"],
        bib_branches=["three", "four"],
        qcd_branches=["five", "six"],
        llp_mH=125,
        llp_mS=100,
    )

    convert_divert(config)

    # Make sure the log message contains something about zero events.
    assert "0 events" in caplog.text


def test_missing_root_files(caplog):
    "run translation, make sure no crash occurs when a empty file is looked at"

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/short_divert_analysis_file.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
            )
        ],
        output_path=Path("./out"),
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
