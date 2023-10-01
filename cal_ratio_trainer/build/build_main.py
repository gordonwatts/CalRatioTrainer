import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from cal_ratio_trainer.config import BuildMainTrainingConfig


def pre_process(df: pd.DataFrame, min_pT: float, max_pT: float):
    # Felix has code that makes it so all of them have the same amount of jets
    # this should be added in for another function
    # And this particular version was copied from Alex.

    # Now For Muon Segments
    logging.debug("Pre-processing Muon Segments")

    # Get all eta Position columns
    filter_MSeg_eta = [
        col for col in df if col.startswith("MSeg_etaPos")  # type: ignore
    ]
    # Get all phi Position columns
    filter_MSeg_phi = [
        col for col in df if col.startswith("MSeg_phiPos")  # type: ignore
    ]
    # Get all phi Direction  columns
    filter_MSeg_phiDir = [
        col for col in df if col.startswith("MSeg_phiDir")  # type: ignore
    ]

    # Subtract the eta of the jet from all MSegs
    df.loc[:, filter_MSeg_eta] = df[filter_MSeg_eta].sub(df["jet_eta"], axis="index")

    # Subtract the phi of the jet from all MSegs
    df.loc[:, filter_MSeg_phi] = df[filter_MSeg_phi].sub(df["jet_phi"], axis="index")

    # Subtract the phi of the jet from all MSegs Dir
    df.loc[:, filter_MSeg_phiDir] = df[filter_MSeg_phiDir].sub(
        df["jet_phi"], axis="index"
    )

    logging.debug(f"Pre-processing jets for {min_pT} GeV < pT < {max_pT} GeV")

    # SCALE JET PT
    df.loc[:, "jet_pT"] = df["jet_pT"].sub(float(min_pT), axis="index")  # type: ignore
    df.loc[:, "jet_pT"] = df["jet_pT"].divide(  # type: ignore
        (float(max_pT) - float(min_pT)), axis="index"
    )

    logging.debug("Pre-processing clusters")
    # DO PHI, ETA Shift

    # Get all eta columns
    filter_clus_eta = [col for col in df if col.startswith("clus_eta")]  # type: ignore
    # Get all phi columns
    filter_clus_phi = [col for col in df if col.startswith("clus_phi")]  # type: ignore
    # Get all pT columns
    filter_clus_pT = [col for col in df if col.startswith("clus_pt")]  # type: ignore

    # Subtract the eta of first cluster(largest pT) from all other
    df.loc[:, filter_clus_eta] = df[filter_clus_eta].sub(df["clus_eta_0"], axis="index")

    # Subtract the phi of first cluster(largest pT) from all other
    # TODO: This is a phi wrap-around bug!
    df.loc[:, filter_clus_phi] = df[filter_clus_phi].sub(df["clus_phi_0"], axis="index")

    # Do eta, phi FLIP

    # Add all etas weighted by pT, then make column that is 1 if positive, -1 if
    # negative
    cluster_sign = np.sum(  # type: ignore
        np.multiply(
            df[filter_clus_eta].fillna(0).to_numpy(),
            df[filter_clus_pT].fillna(0).to_numpy(),
        ),
        axis=1,
    )
    cluster_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(cluster_sign)

    # Flip (multiply by -1) according to previously calculated column
    df.loc[:, filter_clus_eta] = df[filter_clus_eta].multiply(
        cluster_sign, axis="index"
    )

    # SCALE CLUSTER PT
    df.loc[:, filter_clus_pT] = df[filter_clus_pT].sub(min_pT, axis="index")
    df.loc[:, filter_clus_pT] = df[filter_clus_pT].divide(
        (max_pT * 1000 - min_pT), axis="index"
    )

    logging.debug("Pre-processing cluster energy fraction")

    # SCALE Cluster Energy Fraction, then unites layers across different eta ranges
    for i in range(0, 30):
        filter_clus_eFrac = [
            col
            for col in df
            if col.startswith("clus_l") and col.endswith(f"_{i}")  # type: ignore
        ]
        sum_eFrac = df[filter_clus_eFrac].sum(axis=1)  # type: ignore

        df.loc[:, "clus_l1ecal_" + str(i)] = df["clus_l1ecal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l2ecal_" + str(i)] = df["clus_l2ecal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l3ecal_" + str(i)] = df["clus_l3ecal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l4ecal_" + str(i)] = df["clus_l4ecal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )

        df.loc[:, "clus_l1hcal_" + str(i)] = df["clus_l1hcal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l2hcal_" + str(i)] = df["clus_l2hcal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l3hcal_" + str(i)] = df["clus_l3hcal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )
        df.loc[:, "clus_l4hcal_" + str(i)] = df["clus_l4hcal_" + str(i)].divide(
            sum_eFrac, axis="index"
        )

    # Now For Tracks
    logging.debug("Pre-processing tracks")

    # Get all eta columns
    filter_track_eta = [
        col for col in df if col.startswith("track_eta")  # type: ignore
    ]
    # Get all phi columns
    filter_track_phi = [
        col for col in df if col.startswith("track_phi")  # type: ignore
    ]
    # Get all pT columns
    filter_track_pt = [col for col in df if col.startswith("track_pT")]  # type: ignore
    # Get all z vertex columns
    # filter_track_vertex_z = [
    #     col for col in df if col.startswith("track_vertex_z")  # type: ignore
    # ]

    # Subtract the eta of the jet from all tracks
    df.loc[:, filter_track_eta] = df[filter_track_eta].sub(df["jet_eta"], axis="index")

    # Subtract the phi of the jet from all tracks
    df.loc[:, filter_track_phi] = df[filter_track_phi].sub(df["jet_phi"], axis="index")

    # Do eta, phi FLIP

    # SCALE Track PT
    df.loc[:, filter_track_pt] = df[filter_track_pt].sub(min_pT, axis="index")
    df.loc[:, filter_track_pt] = df[filter_track_pt].divide(
        (max_pT - min_pT), axis="index"
    )
    # print(df[filter_track_pt])

    # df[filter_track_vertex_z] = df[filter_track_vertex_z].divide( (100), axis='index')
    # print("1")

    # SCALE Track z0
    filter_track_z0 = [col for col in df if col.startswith("track_z0")]  # type: ignore
    df.loc[:, filter_track_z0] = df[filter_track_z0].divide(250, axis="index")
    # print("2")

    # Add all etas weighted by pT, then make column that is 1 if positive, -1 if
    # negative
    df.loc[:, "track_sign"] = np.sum(  # type: ignore
        np.multiply(
            df[filter_track_eta].fillna(0).to_numpy(),
            df[filter_track_pt].fillna(0).to_numpy(),
        ),
        axis=1,
    )
    df.loc[:, "track_sign"] = df["track_sign"].apply(
        lambda x: 1 * (x >= 0) + (-1) * (x < 0)
    )
    # print("3")

    # Flip (multiply by -1) according to previously calculated column
    df.loc[:, filter_track_eta] = df[filter_track_eta].multiply(
        df["track_sign"], axis="index"
    )
    # print("4")


def split_path_by_wild(p: Path) -> Tuple[Path, Optional[Path]]:
    """Split a path by the last wildcard character, respecting directory boundaries"""
    # Stop the recursion when we get to the bottom.
    if p == Path(".") or p == Path("/"):
        return p, None

    # Try to go down a level.
    good_path, wild_string = split_path_by_wild(p.parent)

    if wild_string is not None:
        # We found a wildcard string, so we need to just add the name on here.
        return good_path, wild_string / p.name

    # If we get here, we didn't find a wildcard string, so we need to check if
    # this path has a wildcard in it. The wildcard can be "*", "?", "[", or "]".
    if "?" in p.name or "[" in p.name or "]" in p.name or "*" in p.name:
        return good_path, Path(p.name)
    else:
        return good_path / p.name, None


def build_main_training(config: BuildMainTrainingConfig):
    """Build main training file."""
    # Load up all the DataFrames and concat them into a single dataframe.

    assert config.min_jet_pT is not None, "No min jet pT specified"
    assert config.max_jet_pT is not None, "No max jet pT specified"

    df: Optional[pd.DataFrame] = None
    assert config.input_files is not None, "No input files specified"

    for f_info in config.input_files:
        file_df: Optional[pd.DataFrame] = None

        # Use the f_info.input_file as a "glob" expression and loop over all found
        # files. Since parent directories might contain the glob character, we need to
        # scan back to the longest root that contains no wildcard characters and then
        # pass the remaining to `glob` based on the preceding valid string created as
        # a `Path` object. For example, if we have a file path of
        # `/foo/bar/baz*/*.pkl`, we need to scan back to `/foo/bar` and then pass
        # `baz*/*.pkl` to `glob`.
        stable, wild = split_path_by_wild(f_info.input_file)
        logging.debug(f'Found stable path "{stable}" and wildcard "{wild}"')
        files_found = [stable] if wild is None else stable.glob(str(wild))

        for count, f_name in enumerate(files_found):
            logging.debug(f'  Processing file #{count+1}: "{f_name}"')
            next_df = pd.read_pickle(f_name)
            assert (
                next_df is not None
            ), f"Unable to read input files {f_info.input_file}"

            # Top level global filters
            if f_info.event_filter is not None:
                # Use the python engine, which is slower, because
                # otherwise the `numexpr` tries to convert `uint65` to
                # `int64` and that pops an exception.
                next_df = next_df.query(
                    f_info.event_filter, engine="python"
                )  # type: ignore

            # Next, run the preprocessing on just this file.
            if len(next_df) > 0:
                pre_process(next_df, config.min_jet_pT, config.max_jet_pT)

                # Now, concat it.
                if file_df is None:
                    file_df = next_df
                else:
                    file_df = pd.concat([file_df, next_df])

        # If we are limited, resample randomly. And append to the
        # master training file.
        assert file_df is not None, "No input events found"
        if f_info.num_events is not None:
            if len(file_df) > f_info.num_events:
                file_df = file_df.sample(f_info.num_events)
            else:
                logging.warning(
                    f"File {f_info.input_file}: Requested {f_info.num_events} events,"
                    f" but only {len(file_df)} available. Ignoring limit."
                )

        if df is None:
            df = file_df
        else:
            df = pd.concat([df, file_df])
            logging.debug(f"  Total events in cumulative dataframe is: {len(df)}")

    assert df is not None

    # Write it out.
    assert config.output_file is not None, "No output file specified"
    df.to_pickle(config.output_file)
