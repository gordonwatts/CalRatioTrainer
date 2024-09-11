import logging
from pathlib import Path
import shutil
import pandas as pd
from cal_ratio_trainer.build.build_main import build_main_training, split_path_by_wild
from cal_ratio_trainer.config import BuildMainTrainingConfig, training_input_file


def test_build_main_one_file_old(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76

    assert any(
        "old input file" in line for line in caplog.text.splitlines()
    ), f"Failed: {caplog.text}"

    # Check for bogus delta phi's
    assert len(df[abs(df.clus_phi_1) > 3.14]) == 0
    assert len(df[abs(df.track_phi_1) > 3.14]) == 0


def test_bad_build_file(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    bad_file = tmp_path / "files_to_get" / "bad.pkl.lock"
    bad_file.parent.mkdir()
    bad_file.touch()
    shutil.copyfile("tests/data/sig_311424_600_275.pkl", bad_file.parent / "signal.pkl")
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            ),
            training_input_file(
                input_file=Path(f"{bad_file.parent}/*"), num_events=None
            ),
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76 * 2

    assert any(
        "old input file" in line for line in caplog.text.splitlines()
    ), f"Failed: {caplog.text}"


def test_build_main_one_file_new(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311421_600_50_new.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 800

    assert len(caplog.text) == 0

    # And make sure they aren't too large
    assert len(df[abs(df.clus_phi_1) > 0.4]) == 0
    assert len(df[abs(df.track_phi_1) > 0.4]) == 0


def test_build_clus_pt_0(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    # TODO: if we reverse sorting, then this perhaps should be 0 not 6.
    assert len(df[df.clus_pt_0 > 0.1]) > 1


def test_build_sig_has_llp_columns(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)

    assert "llp_mH" in df.columns
    assert "llp_mS" in df.columns


def test_build_zero_length_file(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/one_length_file.pkl"),
                num_events=None,
                event_filter="eventNumber % 2 == 0",
            ),
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 0


def test_build_main_one_file_ask_for_too_much(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=500
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert "Requested 500 events, but only 76 available" in caplog.text


def test_build_main_one_file_twice(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            ),
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            ),
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76 * 2


def test_build_main_one_file_length(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=2
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 2


def test_build_via_wildcard(tmp_path):
    out_file = tmp_path / "output" / "test_output.pkl"
    out_file.parent.mkdir()

    # Build directory with several files in it.
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()

    # write out the files we want
    for i in range(10):
        in_file = in_dir / f"sig_311424_600_275_{i}.pkl"
        in_file.write_bytes((Path("tests/data/sig_311424_600_275.pkl")).read_bytes())

    # Write out the files that should be bogus
    for i in range(10):
        # copy the file from tests/data/sig_311424_600_275.pkl over to in_dir:
        in_file = in_dir / f"fork_311424_600_275_{i}.pkl"
        in_file.write_bytes((Path("tests/data/sig_311424_600_275.pkl")).read_bytes())

    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(input_file=Path(f"{in_dir}/sig*.pkl"), num_events=None)
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76 * 10


def test_build_good_warning_if_no_files(tmp_path):
    out_file = tmp_path / "output" / "test_output.pkl"
    out_file.parent.mkdir()

    # Build directory with several files in it.
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()

    try:
        c = BuildMainTrainingConfig(
            input_files=[
                training_input_file(
                    input_file=Path(f"{in_dir}/sig*.pkl"), num_events=None
                )
            ],
            output_file=out_file,
            min_jet_pt=30,
            max_jet_pt=400,
        )
        build_main_training(c)

        assert False

    except ValueError as e:
        assert "sig*.pkl" in str(e)


def test_build_via_wildcard_with_limit(tmp_path):
    out_file = tmp_path / "output" / "test_output.pkl"
    out_file.parent.mkdir()

    # Build directory with several files in it.
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()

    # write out the files we want
    for i in range(10):
        in_file = in_dir / f"sig_311424_600_275_{i}.pkl"
        in_file.write_bytes((Path("tests/data/sig_311424_600_275.pkl")).read_bytes())

    # Write out the files that should be bogus
    for i in range(10):
        # copy the file from tests/data/sig_311424_600_275.pkl over to in_dir:
        in_file = in_dir / f"fork_311424_600_275_{i}.pkl"
        in_file.write_bytes((Path("tests/data/sig_311424_600_275.pkl")).read_bytes())

    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(input_file=Path(f"{in_dir}/sig*.pkl"), num_events=100)
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 100


def test_build_via_nested_wildcard(tmp_path):
    out_file = tmp_path / "output" / "test_output.pkl"
    out_file.parent.mkdir()

    # Build directory with several files in it.
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()

    # write out the files we want
    for i in range(10):
        nested_dir = in_dir / f"subdir_{i}"
        nested_dir.mkdir()
        in_file = nested_dir / f"sig_311424_600_275_{i}.pkl"
        in_file.write_bytes((Path("tests/data/sig_311424_600_275.pkl")).read_bytes())

    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path(f"{in_dir}/*/sig*.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76 * 10


def test_split_by_wild():
    assert split_path_by_wild(Path("foo/bar/baz")) == (Path("foo/bar/baz"), None)
    assert split_path_by_wild(Path("/foo/bar/baz")) == (Path("/foo/bar/baz"), None)
    assert split_path_by_wild(Path("foo/bar/baz/*.pkl")) == (
        Path("foo/bar/baz"),
        Path("*.pkl"),
    )
    assert split_path_by_wild(Path("foo/bar/*baz*/*.pkl")) == (
        Path("foo/bar"),
        Path("*baz*/*.pkl"),
    )


def test_include_only_odd_events(tmp_path):
    "Test the event filter expression"
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"),
                num_events=None,
                event_filter="eventNumber % 2 == 1",
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df[df.eventNumber % 2 == 0]) == 0
    assert len(df) >= 1


def test_include_bib_file_with_expression(tmp_path):
    "Crash seen in the real world"
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/bib.pkl"),
                num_events=None,
                event_filter="eventNumber % 2 == 1",
            )
        ],
        output_file=out_file,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df[df.eventNumber % 2 == 0]) == 0
    assert len(df) >= 1


def test_drop_columns(tmp_path):
    "Crash seen in the real world"
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/bib.pkl"),
                num_events=None,
                event_filter="eventNumber % 2 == 1",
            )
        ],
        output_file=out_file,
        min_jet_pt=40,
        max_jet_pt=500,
        remove_branches=["HLT_jet_eta", "bogus_no_problem"],
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)

    assert "HLT_jet_eta" not in df.columns


def test_include_bib_file(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/bib.pkl"),
            )
        ],
        output_file=out_file,
        min_jet_pt=40,
        max_jet_pt=500,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 558


def test_build_no_copy_view_errors(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # logging.warning("hi")

    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"),
                num_events=None,
            )
        ],
        output_file=out_file,
        min_jet_pt=30,
        max_jet_pt=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76

    assert "SettingWithCopyWarning" not in caplog.text
