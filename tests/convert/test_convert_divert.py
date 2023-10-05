from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from cal_ratio_trainer.common.file_lock import FileLock
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
                llp_mH=125,
                llp_mS=100,
            )
        ],
        output_path=tmp_path,
        signal_branches=["one", "two"],
        bib_branches=["three", "four"],
        qcd_branches=["five", "six"],
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
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=["one"],
        bib_branches=["one"],
        qcd_branches=["one"],
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
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
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


def test_qcd_file(caplog, tmp_path):
    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/short_divert_analysis_file.root"),
                data_type=DiVertFileType.qcd,
                output_dir=None,
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "short_divert_analysis_file.pkl"
    df = pd.read_pickle(output_file)

    assert df.dtypes["llp_mS"] == "float64"
    assert df.dtypes["llp_mH"] == "float64"
    assert df.dtypes["label"] == "int64"


def test_qcd_multi_jets(caplog, tmp_path):
    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311424_600_275.root"),
                data_type=DiVertFileType.qcd,
                output_dir=None,
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df.eventNumber.unique()) < len(df)

    def match_event_and_cluster_pt(event_number: int, jet_pt: float, clus_pt: float):
        tree_data = uproot.open("tests/data/sig_311424_600_275.root")[
            "trees_DV_"
        ].arrays()  # type: ignore
        event = tree_data[tree_data.eventNumber == event_number]
        assert len(event) == 1, f"Event {event_number} not found"

        # get the jet index for this jet pt.
        jet_index_list = event.nn_jet_index[event.jet_pT == jet_pt]
        assert len(jet_index_list) == 1, f"Jet {jet_pt} not found"
        jet_index = jet_index_list[0][0]

        clus_list_mask = event.clus_pt == clus_pt
        assert (
            ak.sum(clus_list_mask) == 1
        ), f"Cluster {clus_pt} not found in the cluster list for the event!"
        clus_jet_index = event.cluster_jetIndex[clus_list_mask]
        assert len(clus_jet_index) == 1, f"Cluster {clus_pt} not found"
        assert jet_index in clus_jet_index[0], (
            f"Cluster {clus_pt:.2f} not part of clusters for jet {jet_index} - looks like"
            f" it is from {clus_jet_index[0][0]}"
        )

    match_event_and_cluster_pt(df.eventNumber[0], df.jet_pT[0], df.clus_pt_1[0])
    match_event_and_cluster_pt(df.eventNumber[0], df.jet_pT[0], df.clus_pt_0[0])
    match_event_and_cluster_pt(df.eventNumber[1], df.jet_pT[1], df.clus_pt_0[1])


def test_bib_file(tmp_path, caplog):
    "Second run, and nothing should happen"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/bib.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "bib.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) == 1

    assert df.dtypes["llp_mS"] == "float64"
    assert df.dtypes["llp_mH"] == "float64"
    assert df.dtypes["label"] == "int64"

    assert "HLT_jet_isBIB" not in df.columns


def test_lock_file(tmp_path, caplog):
    "Lock file causes skip"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/bib.root"),
                data_type=DiVertFileType.bib,
                output_dir=None,
                llp_mH=0,
                llp_mS=0,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    # Create lock file.
    output_file = tmp_path / "bib.pkl"
    with FileLock(output_file) as lock:
        assert lock.is_locked

        # Run
        convert_divert(config)

        # Make sure there is no output file.
        assert not output_file.exists()

        # Make sure a warning was issued
        assert "skipped" not in caplog.text


def test_sig_file(tmp_path, caplog):
    "Make sure a signal file runs correctly"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311424_600_275.root"),
                data_type=DiVertFileType.sig,
                output_dir=None,
                llp_mH=600,
                llp_mS=275,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) == 91
    assert "llp_mS" in df.columns
    assert "llp_mH" in df.columns

    assert df["llp_mS"].unique() == [275]
    assert df["llp_mH"].unique() == [600]

    assert df.dtypes["llp_mS"] == "float64"
    assert df.dtypes["llp_mH"] == "float64"
    assert df.dtypes["label"] == "int64"


def test_sig_eta(tmp_path):
    "Make sure we aren't cutting eta tightly"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311424_600_275.root"),
                data_type=DiVertFileType.sig,
                output_dir=None,
                llp_mH=600,
                llp_mS=275,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df[abs(df.jet_eta) > 2.0]) > 0


def test_cluster_pt(tmp_path):
    "Make sure we aren't cutting eta tightly"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311424_600_275.root"),
                data_type=DiVertFileType.sig,
                output_dir=None,
                llp_mH=600,
                llp_mS=275,
            )
        ],
        output_path=tmp_path,
        signal_branches=default_branches.signal_branches,
        bib_branches=default_branches.bib_branches,
        qcd_branches=default_branches.qcd_branches,
    )

    convert_divert(config)

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df[df.clus_pt_0 > 0]) > 0


# def test_pt_sorting(tmp_path):
