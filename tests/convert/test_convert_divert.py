from pathlib import Path

import awkward as ak
import numpy
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
from cal_ratio_trainer.common.column_names import (
    col_cluster_track_mseg_names,
    col_jet_names,
    col_llp_mass_names,
    event_level_names,
)


def assert_columns(df: pd.DataFrame):
    "Make sure all the columns are in the DataFrame that we expect"

    # Build a list of all columns
    all_column_names = (
        col_cluster_track_mseg_names
        + col_jet_names
        + col_llp_mass_names
        + event_level_names
    )

    # Build a list of those column names not in the DataFrame.
    missing_columns = [name for name in all_column_names if name not in df.columns]
    assert len(missing_columns) == 0, f"Missing columns {missing_columns}"


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
        rename_branches={"nn_jet_index": "jet_index"},
        min_jet_pt=40,
        max_jet_pt=500,
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
        rename_branches={"nn_jet_index": "jet_index"},
        min_jet_pt=40,
        max_jet_pt=500,
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "short_divert_analysis_file.pkl"
    df = pd.read_pickle(output_file)

    assert df.dtypes["llp_mS"] == "float64"
    assert df.dtypes["llp_mH"] == "float64"
    assert df.dtypes["label"] == "int64"

    assert_columns(df)


def test_qcd_bad_slice(caplog, tmp_path):
    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/qcd_slice_error.root"),
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "qcd_slice_error.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) > 0


def test_jet_cuts(caplog, tmp_path):
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=25,
        max_jet_pt=100,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "short_divert_analysis_file.pkl"
    df = pd.read_pickle(output_file)

    assert not any(df.jet_pt < 25)
    assert not any(df.jet_pt >= 100)


def test_sig_as_qcd_file(caplog, tmp_path):
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "sig_311424_600_275.pkl"
    assert output_file.exists()


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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    # Find the output file
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df.eventNumber.unique()) < len(df)

    # Code to help with dipping into the event data structure and making sure,
    # by hand, we've built up a good set of clusters and tracks and MSeg's.
    def match_event_and_cluster_pt(event_number: int, jet_pt: float, clus_pt: float):
        match_event_and_common(
            event_number, jet_pt, clus_pt, "clus_pt", "cluster_jetIndex", "cluster", 1
        )

    def match_event_and_track_pt(event_number: int, jet_pt: float, track_pt: float):
        match_event_and_common(
            event_number, jet_pt, track_pt, "track_pT", "track_jetIndex", "track", 2
        )

    def match_event_and_MSeg_etaDir(event_number: int, jet_pt: float, eta_dir: float):
        match_event_and_common(
            event_number, jet_pt, eta_dir, "MSeg_etaDir", "MSeg_jetIndex", "MSeg", 2
        )

    def match_event_and_common(
        event_number: int,
        jet_pt: float,
        item: float,
        event_item_name: str,
        event_item_jetIndex: str,
        name: str,
        deref_times: int,
    ):
        tree_data = uproot.open("tests/data/sig_311424_600_275.root")[
            "trees_DV_"
        ].arrays()  # type: ignore
        event = tree_data[tree_data.eventNumber == event_number]
        assert len(event) == 1, f"Event {event_number} not found"

        # get the jet index for this jet pt.
        jet_index_list = event.nn_jet_index[event.jet_pT == jet_pt]
        assert len(jet_index_list) == 1, f"Jet {jet_pt} not found"
        jet_index = jet_index_list[0][0]

        # Now, find the item in the list of clusters, tracks, etc.
        clus_list_mask = event[event_item_name] == item
        assert (
            ak.sum(clus_list_mask) == 1
        ), f"{name} {item} not found in the {event_item_name} for the event!"

        # Now find the matching index. This depends if the list is nested
        # or not. For clusters it is nested once, for tracks and MSegs it is
        # nested twice (e.g. each can be attached to more than one jet).
        item_jet_index_list = event[event_item_jetIndex][clus_list_mask]
        if deref_times == 2:
            assert (
                len(item_jet_index_list) == 1
                and len(item_jet_index_list[0]) == 1
                and len(item_jet_index_list[0][0]) == 1
            ), f"Item {item} not found in the jet match list"
            item_jet_index = item_jet_index_list[0][0]
        elif deref_times == 1:
            assert (
                len(item_jet_index_list) == 1 and len(item_jet_index_list[0]) == 1
            ), f"Item {item} not found in the jet match list"
            item_jet_index = item_jet_index_list[0]
        else:
            raise RuntimeError(f"Unknown deref times {deref_times}")

        assert jet_index in item_jet_index, (
            f"{name} {item:.2f} not part of {name}s for jet {jet_index} - looks "
            f"like it associated to jet(s) {item_jet_index}"
        )

    match_event_and_cluster_pt(df.eventNumber[0], df.jet_pt[0], df.clus_pt_1[0])
    match_event_and_cluster_pt(df.eventNumber[0], df.jet_pt[0], df.clus_pt_0[0])
    match_event_and_cluster_pt(df.eventNumber[1], df.jet_pt[1], df.clus_pt_0[1])
    match_event_and_track_pt(df.eventNumber[0], df.jet_pt[0], df.track_pt_0[0])
    match_event_and_MSeg_etaDir(df.eventNumber[0], df.jet_pt[0], df.MSeg_etaDir_0[0])


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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
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

    assert_columns(df)


def test_bib_slice_error(tmp_path, caplog):
    "A slice error crash found in the wild"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/bib_25563461_slice_error.root"),
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "0 events" in caplog.text

    # Check what was written out.
    output_file = tmp_path / "bib_25563461_slice_error.pkl"
    assert not output_file.exists()


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
        rename_branches={"nn_jet_index": "jet_index"},
        min_jet_pt=40,
        max_jet_pt=500,
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) == 101
    assert "llp_mS" in df.columns
    assert "llp_mH" in df.columns

    assert df["llp_mS"].unique() == [275]
    assert df["llp_mH"].unique() == [600]

    assert df.dtypes["llp_mS"] == "float64"
    assert df.dtypes["llp_mH"] == "float64"
    assert df.dtypes["label"] == "int64"

    # Make sure nothing is too far away for llp matching.
    jets = ak.zip(
        {
            "eta": df.jet_eta,
            "phi": df.jet_phi,
            "pt": df.jet_pt,
        },
        with_name="Momentum3D",
    )
    llps = ak.zip(
        {
            "eta": df.llp_eta,
            "phi": df.llp_phi,
            "pt": df.llp_pt,
        },
        with_name="Momentum3D",
    )

    dR = jets.deltaR(llps)  # type: ignore

    assert not ak.any(dR > 0.4)

    # Make sure we have some real negative eta's
    assert numpy.any(df.jet_eta < -1.5)

    assert_columns(df)


def test_sig_file_bomb(tmp_path, caplog):
    "Make sure a signal file runs correctly"

    default_branches = load_config(ConvertDiVertAnalysisConfig)

    config = ConvertDiVertAnalysisConfig(
        input_files=[
            DiVertAnalysisInputFile(
                input_file=Path("tests/data/sig_311314_bad_object_small.root"),
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    # Check what was written out.
    output_file = tmp_path / "sig_311314_bad_object_small.pkl"
    df = pd.read_pickle(output_file)

    assert len(df) > 0


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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
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
        rename_branches=default_branches.rename_branches,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    convert_divert(config)

    # Check what was written out.
    output_file = tmp_path / "sig_311424_600_275.pkl"
    df = pd.read_pickle(output_file)

    assert len(df[df.clus_pt_0 > 0]) > 0
