import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import numpy as np
import pandas as pd
from cal_ratio_trainer.config import BuildMainTrainingConfig
import dask.dataframe as dd
import dask

from cal_ratio_trainer.common.column_names import all_cols


def pre_process(df: pd.DataFrame, min_pT: float, max_pT: float):
    # Felix has code that makes it so all of them have the same amount of jets
    # this should be added in for another function
    # And this particular version was copied from Alex.

    # Check to see if this is zero length.
    if len(df) == 0:
        return df

    logging.debug(f"Rescaling jets for {min_pT} GeV < pT < {max_pT} GeV")

    # SCALE JET PT
    df.loc[:, "jet_pt"] = df["jet_pt"].sub(float(min_pT), axis="index")  # type: ignore
    df.loc[:, "jet_pt"] = df["jet_pt"].divide(  # type: ignore
        (float(max_pT) - float(min_pT)), axis="index"
    )

    logging.debug("Pre-processing clusters")
    # DO PHI, ETA Shift

    # Get all eta columns
    filter_clus_eta = [col for col in df if col.startswith("clus_eta")]  # type: ignore

    # Get all pT columns
    filter_clus_pT = [col for col in df if col.startswith("clus_pt")]  # type: ignore

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

    # Scale cluster pT by the total cluster pT's
    # TODO: use row / row.sum(), which rescaled each guy, or
    # rescale by the upper limit (which is what is currently being done)?
    # Note that we need to rescale here because we are in MeV, not GeV like
    # everything else here. However, as far as I can tell, we never
    # correct for this unit problem in the DiVertAnalysis code.
    # def rescale_row_by_sum(row):
    #     return row / row.sum()

    # df.loc[:, filter_clus_pT] = df[filter_clus_pT].apply(rescale_row_by_sum, axis=1)
    df.loc[:, filter_clus_pT] = df[filter_clus_pT].sub(min_pT, axis="index")
    df.loc[:, filter_clus_pT] = df[filter_clus_pT].divide(
        (max_pT - min_pT) * 1000.0, axis="index"
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

    # Get all pT columns
    filter_track_pt = [col for col in df if col.startswith("track_pt")]  # type: ignore

    # Do eta, phi FLIP

    # SCALE Track PT
    df.loc[:, filter_track_pt] = df[filter_track_pt].sub(min_pT, axis="index")
    df.loc[:, filter_track_pt] = df[filter_track_pt].divide(
        (max_pT - min_pT), axis="index"
    )

    # df[filter_track_vertex_z] = df[filter_track_vertex_z].divide( (100), axis='index')

    # SCALE Track z0
    filter_track_z0 = [col for col in df if col.startswith("track_z0")]  # type: ignore
    df.loc[:, filter_track_z0] = df[filter_track_z0].divide(250, axis="index")

    # Add all etas weighted by pT, then make column that is 1 if positive, -1 if
    # negative
    track_sign = np.sum(  # type: ignore
        np.multiply(
            df[filter_track_eta].fillna(0).to_numpy(),
            df[filter_track_pt].fillna(0).to_numpy(),
        ),
        axis=1,
    )
    track_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(track_sign)

    # Flip (multiply by -1) according to previously calculated column
    df.loc[:, filter_track_eta] = df[filter_track_eta].multiply(
        track_sign, axis="index"
    )

    return df


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


def pickle_loader(drop_branches: Optional[List[str]]) -> Callable[[Path], pd.DataFrame]:
    "Create a function that will drop some branches!"

    def my_pickle_loader(f: Path) -> pd.DataFrame:
        """Read the file, and make sure that we have the right columns.

        Args:
            f (Path): file to load

        Returns:
            pd.DataFrame: Sanitized DataFrame
        """
        df = pd.read_pickle(f)  # type: pd.DataFrame

        # checking if the dataframe is empty
        if len(df) == 0:
            return df

        # Rename any columns
        # TODO: Remove this code once understand how this happened upstream.
        # See issue https://github.com/gordonwatts/CalRatioTrainer/issues/116
        if "mH" in df.columns:
            logging.warning(
                "Renaming mH to llp_mH - should not need to happen - old input file?"
            )
            df.rename(columns={"mH": "llp_mH"}, inplace=True)
        if "mS" in df.columns:
            logging.warning(
                "Renaming mS to llp_mS - should not need to happen - old input file?"
            )
            df.rename(columns={"mS": "llp_mS"}, inplace=True)

        df.rename(
            columns={
                col: col.replace("_pT", "_pt")
                for col in df.columns
                if col.endswith("_pT")
            },
            inplace=True,
        )
        df.rename(
            columns={
                col: col.replace("_pT_", "_pt_") for col in df.columns if "_pT_" in col
            },
            inplace=True,
        )

        df = df[all_cols]

        # Get rid of branches that should not, perhaps, have been
        # written out in the first place!
        if drop_branches is not None:
            for b in drop_branches:
                if b in df.columns:
                    logging.warning(
                        f"Dropping branch {b} - should not have been "
                        "written in first place?"
                    )
                    df.drop(columns=b, inplace=True)

        return df

    return my_pickle_loader


def build_main_training(config: BuildMainTrainingConfig):
    """Build main training file."""
    # Load up all the DataFrames and concat them into a single dataframe.

    assert config.min_jet_pt is not None, "No min jet pT specified"
    assert config.max_jet_pt is not None, "No max jet pT specified"

    df: Optional[pd.DataFrame] = None
    assert config.input_files is not None, "No input files specified"

    for f_info in config.input_files:
        # Use the f_info.input_file as a "glob" expression and loop over all found
        # files. Since parent directories might contain the glob character, we need to
        # scan back to the longest root that contains no wildcard characters and then
        # pass the remaining to `glob` based on the preceding valid string created as
        # a `Path` object. For example, if we have a file path of
        # `/foo/bar/baz*/*.pkl`, we need to scan back to `/foo/bar` and then pass
        # `baz*/*.pkl` to `glob`.
        stable, wild = split_path_by_wild(f_info.input_file)
        logging.debug(f'Found stable path "{stable}" and wildcard "{wild}"')
        files_found_all = [stable] if wild is None else list(stable.glob(str(wild)))
        files_found = [f for f in files_found_all if f.is_file() and f.suffix == ".pkl"]

        if len(files_found) == 0:
            raise ValueError(
                f"No files found for input file/pattern {f_info.input_file}"
            )

        ddf = dd.from_delayed(  # type: ignore
            [
                dask.delayed(pickle_loader(config.remove_branches))(  # type: ignore
                    f_name
                )
                for f_name in files_found
            ]
        )

        # Top level global filter for events
        if f_info.event_filter is not None:
            ddf = ddf.query(f_info.event_filter, engine="python")  # type: ignore

        # Do the preprocessing
        processed_ddf = ddf.map_partitions(  # type: ignore
            lambda df: pre_process(
                df, config.min_jet_pt, config.max_jet_pt  # type: ignore
            )
        )  # type: List

        # Resample the thing
        if f_info.num_events is not None:
            ddf_len = len(processed_ddf)
            if ddf_len > f_info.num_events:
                fraction = f_info.num_events / ddf_len
                processed_ddf = processed_ddf.sample(frac=fraction)  # type: ignore
            else:
                logging.warning(
                    f"File {f_info.input_file}: Requested {f_info.num_events} events,"
                    f" but only {len(processed_ddf)} available. Ignoring limit."
                )

        file_df = processed_ddf.compute()  # type: ignore

        df = file_df if df is None else pd.concat([df, file_df])
        logging.debug(f"  Total events in cumulative dataframe is: {len(df)}")

    assert df is not None, "Failed to add any events to the overall DataFrame"

    # Write it out.
    assert config.output_file is not None, "No output file specified"
    df.to_pickle(config.output_file)
