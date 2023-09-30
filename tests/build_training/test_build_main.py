from pathlib import Path
import pandas as pd
from cal_ratio_trainer.build.build_main import build_main_training, split_path_by_wild
from cal_ratio_trainer.config import BuildMainTrainingConfig, training_input_file


def test_build_main_one_file(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pT=30,
        max_jet_pT=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76


def test_build_main_one_file_ask_for_too_much(tmp_path, caplog):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=500
            )
        ],
        output_file=out_file,
        min_jet_pT=30,
        max_jet_pT=400,
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
        min_jet_pT=30,
        max_jet_pT=400,
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
        min_jet_pT=30,
        max_jet_pT=400,
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
        min_jet_pT=30,
        max_jet_pT=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76 * 10


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
        min_jet_pT=30,
        max_jet_pT=400,
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
        min_jet_pT=30,
        max_jet_pT=400,
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
        min_jet_pT=30,
        max_jet_pT=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df[df.eventNumber % 2 == 0]) == 0
    assert len(df) >= 1
