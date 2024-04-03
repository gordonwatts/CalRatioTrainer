import glob
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import awkward as ak
import numpy as np
import pandas as pd
import uproot
import vector
from vector._compute.planar.deltaphi import rectify

from cal_ratio_trainer.common.column_names import EventType
from cal_ratio_trainer.common.file_lock import FileLock
from cal_ratio_trainer.config import ConvertDiVertAnalysisConfig

import time
import cProfile

# Much of the code was copied directly from Alex Golub's code on gitlab.
# Many thanks to their work for this!

vector.register_awkward()


def jets_masking(array: ak.Array, min_jet_pt: float, max_jet_pt: float) -> ak.Array:
    """
    Returns an awkward array containing only good, central jets with pT between 40 and
    500 GeV.

    Parameters:
    -----------
    array : ak.Array
        An awkward array containing jet information.

    Returns:
    --------
    ak.Array
        An awkward array containing only good, central jets with pT between 40 and 500
        GeV.
    """
    logging.debug("Masking jets")
    # Only good, central, jets are considered.
    jet_pT_mask = (array.jets.pt >= min_jet_pt) & (  # type: ignore
        array.jets.pt < max_jet_pt  # type: ignore
    )
    jet_eta_mask = np.abs(array.jets.eta) <= 2.5  # type: ignore

    return array.jets[jet_pT_mask & jet_eta_mask]  # type: ignore


def applying_llp_cuts(llps: ak.Array):
    """
    Applies cuts to the input LLPs (Long-Lived Particles) array and returns only the
    "good" LLPs that pass the basic LLP acceptance.

    Args:
        llps (ak.Array): An array of LLPs with eta, Lxy, and Lz attributes.

    Returns:
        ak.Array: An array of "good" LLPs that pass the cuts.
    """
    logging.debug("Applying LLP fiducial cuts")
    central_llp_eta_mask = abs(llps.eta) < 1.4  # type: ignore
    endcap_llp_eta_mask = abs(llps.eta) > 1.4  # type: ignore

    llp_Lxy_mask = (llps.Lxy > 1200) & (llps.Lxy < 4000)  # type: ignore
    llp_Lz_mask = (llps.Lz > 3500) & (llps.Lz < 6000) | (  # type: ignore
        llps.Lz < -3500
    ) & (
        llps.Lz > -6000
    )  # type: ignore

    llp_mask = (central_llp_eta_mask & llp_Lxy_mask) | (
        endcap_llp_eta_mask & llp_Lz_mask
    )
    return llps[llp_mask]


def metric_table(
    first: ak.Array,
    other: ak.Array,
    axis: Optional[int] = 1,
    metric: Callable[[ak.Array, ak.Array], ak.Array] = lambda a, b: a.deltaR2(
        b
    ),  # type: ignore
    return_combinations: bool = False,
):
    """Return a list of a metric evaluated between first object and another.

    The two arrays should be broadcast-compatible on all axes other than the specified
    axis, which will be used to form a cartesian product. If axis=None, broadcast
    arrays directly. The return shape will be that of ``first`` with a new axis with
    shape of ``other`` appended at the specified axis depths.

    Code copied from coffea: https://github.com/CoffeaTeam/coffea/blob/master/src/
    coffea/nanoevents/methods/vector.py#L675-L710

    Parameters
    ----------
        first : awkward.Array
            First array to base the cartesian product on
        other : awkward.Array
            Another array with same shape in all but ``axis``
        axis : int, optional
            The axis to form the cartesian product (default 1). If None, the metric
            is directly evaluated on the input arrays (i.e. they should broadcast)
        metric : callable
            A function of two arguments, returning a scalar. The default metric is
            `delta_r`.
        return_combinations : bool
            If True return the combinations of inputs as well as an unzipped tuple
    """
    if axis is None:
        a, b = first, other
    else:
        a, b = ak.unzip(ak.cartesian([first, other], axis=axis, nested=True))
    mval = metric(a, b)  # type: ignore
    if return_combinations:
        return mval, (a, b)
    return mval


def nearest(
    first: ak.Array,
    other: ak.Array,
    axis: Optional[int] = 1,
    metric: Callable[[ak.Array, ak.Array], ak.Array] = lambda a, b: a.deltaR(
        b
    ),  # type:ignore
    return_metric: bool = False,
    threshold: Optional[float] = None,
):
    """Return nearest object to this one

    Finds item in ``other`` satisfying ``min(metric(first, other))``.
    The two arrays should be broadcast-compatible on all axes other than the specified
    axis, which will be used to form a cartesian product. If axis=None, broadcast
    arrays directly. The return shape will be that of ``first``.

    Code copied from coffea: https://github.com/CoffeaTeam/coffea/blob/master/src/
    coffea/nanoevents/methods/vector.py#L675-L710

    Parameters
    ----------
        other : awkward.Array
            Another array with same shape in all but ``axis``
        axis : int, optional
            The axis to form the cartesian product (default 1). If None, the metric
            is directly evaluated on the input arrays (i.e. they should broadcast)
        metric : callable
            A function of two arguments, returning a scalar. The default metric is
            `delta_r`.
        return_metric : bool, optional
            If true, return both the closest object and its metric (default false)
        threshold : Number, optional
            If set, any objects with ``metric > threshold`` will be masked from the
            result
    """
    mval, (_, b) = metric_table(  # type: ignore
        first, other, axis, metric, return_combinations=True
    )
    if axis is None:
        # NotImplementedError: awkward.firsts with axis=-1
        axis = other.layout.purelist_depth - 2  # type: ignore
    assert axis is not None
    mmin = ak.argmin(mval, axis=axis + 1, keepdims=True)
    out = ak.firsts(b[mmin], axis=axis + 1)  # type: ignore
    metric = ak.firsts(mval[mmin], axis=axis + 1)  # type: ignore
    if threshold is not None:
        out = out.mask[metric <= threshold]  # type: ignore
    if return_metric:
        return out, metric
    return out


def sort_by_pt(data: ak.Array) -> ak.Array:
    """
    Sorts the clusters and tracks in the input `data` object by their transverse
    momentum (pt), in descending order. Returns a new `Data` object with the
    same content as `data`, but with the clusters and tracks sorted by pt.

    Parameters
    ----------
    data : Data
        The input data object to be sorted.

    Returns
    -------
    Data
        A new `Data` object with the same content as `data`, but with the clusters and
        tracks sorted by pt.
    """
    new_clusters_index = ak.argsort(
        data.clusters.pt,  # type: ignore
        axis=1,
        ascending=False,
    )
    new_clusters = data.clusters[new_clusters_index]  # type: ignore

    new_track_index = ak.argsort(
        data.tracks.pt,  # type: ignore
        axis=1,
        ascending=False,
    )
    new_tracks = data.tracks[new_track_index]  # type: ignore

    # Apparently this is not stable between C++ and python, so have to leave it in
    # the order we found it. The current C++ code does not do ordering either.
    # new_mseg_index = ak.argsort(data.msegs.phiDir, axis=1, ascending=False)
    # new_msegs = data.msegs[new_mseg_index]

    # And re-create the array and return it!
    return remake_by_replacing(
        data, clusters=new_clusters, tracks=new_tracks  # type: ignore
    )


def column_guillotine(data: ak.Array) -> pd.DataFrame:
    """
    Takes an array with clus/mseg/track columns not split
    up and splits them up into new columns.
    Number of new columns hardcoded in
    Mseg_cols/clus_cols/track_cols also hardcoded
    Since those should be the same for qcd/signal/bib events

    # track = 20
    # cluster = 30
    # mseg = 30

    Returns: dataframe with all the new split columns
    (and preexisting old columns that didn't need to be split)
    """
    # Sort everything (clusters, etc)
    sorted_data = sort_by_pt(data)

    # Next, expand them to fill all the input slots for the network training.
    def expand_and_jet_filter(
        jet_index_list: ak.Array,
        to_match_objects: ak.Array,
        matched_object_nested_jet_index: bool,
    ) -> ak.Array:
        """Take the jets, which are in their un-flattened state, and match each jet to
        a to_match_objects by jet index. to_match_objects has an entry per event."""

        matches_one_down = ak.unflatten(
            to_match_objects, np.ones(len(to_match_objects), dtype=int)
        )
        pairs = ak.cartesian({"jet": jet_index_list, "to_match": matches_one_down})
        mask = pairs.jet == pairs.to_match.jetIndex  # type: ignore
        # TODO: we should be able to determine this by looking at the jetIndex - single
        # number or list.
        if matched_object_nested_jet_index:
            mask = ak.any(mask, axis=-1)

        matched_objects = pairs.to_match[mask]  # type: ignore
        return matched_objects

    matched_clusters = expand_and_jet_filter(
        sorted_data.jets.jetIndex,  # type: ignore
        sorted_data.clusters,  # type: ignore
        matched_object_nested_jet_index=False,
    )
    matched_tracks = expand_and_jet_filter(
        sorted_data.jets.jetIndex,  # type: ignore
        sorted_data.tracks,  # type: ignore
        matched_object_nested_jet_index=True,
    )
    matched_msegs = expand_and_jet_filter(
        sorted_data.jets.jetIndex,  # type: ignore
        sorted_data.msegs,  # type: ignore
        matched_object_nested_jet_index=True,
    )

    # Now we have all the bits. Switch to a per-jet view rather than a per-event view.
    jet_list = ak.flatten(sorted_data.jets)
    llp_list = (
        None if "llps" not in sorted_data.fields else ak.flatten(sorted_data.llps)
    )
    cluster_list = ak.flatten(matched_clusters)
    track_list = ak.flatten(matched_tracks)
    mseg_list = ak.flatten(matched_msegs)

    # We also have to deal with the per-event view. Each row has to be
    # duplicated the right number of times.

    event_x_jet = ak.cartesian({"event": sorted_data.event, "jet": sorted_data.jets})
    event_list = ak.flatten(event_x_jet.event)  # type: ignore

    # We need to drop the jetIndex column next - because it sometimes is a list,
    # and padding below makes no real sense with zeros.
    def drop_columns(data: ak.Array, drop_list) -> ak.Array:
        return ak.zip(
            {col: data[col] for col in data.fields if col not in drop_list},
            with_name="Momentum3D",  # type:ignore
        )

    jet_list = drop_columns(jet_list, ["jetIndex"])  # type: ignore
    cluster_list = drop_columns(cluster_list, ["jetIndex"])  # type: ignore
    track_list = drop_columns(track_list, ["jetIndex"])  # type: ignore

    # Reorient objects by some central eta/phi (subtrack them).
    def relative_angle(jets: ak.Array, objects: ak.Array):
        # Modify in-place the eta and phi to be relative to the jet axis.
        # Note the ordering of these is important and must be matched in DiVertAnalysis.
        objects["eta"] = objects.deltaeta(jets)  # type: ignore
        objects["phi"] = objects.deltaphi(jets)  # type: ignore

    # Clusters are done w.r.t. the cluster axis.
    relative_angle(cluster_list[:, 0], cluster_list)  # type: ignore
    # Tracks are done w.r.t. the jet axis
    relative_angle(jet_list, track_list)

    mseg_deta_pos = mseg_list.etaPos - jet_list.eta  # type: ignore
    mseg_list["etaPos"] = mseg_deta_pos  # type: ignore
    mseg_dphi_pos = rectify(np, mseg_list.phiPos - jet_list.phi)  # type: ignore
    mseg_list["phiPos"] = mseg_dphi_pos  # type: ignore
    mseg_dphi_dir = rectify(np, mseg_list.phiDir - jet_list.phi)  # type: ignore
    mseg_list["phiDir"] = mseg_dphi_dir  # type: ignore

    # Next, lets pad, with zeros, the cluster, track, and mseg to the length we are
    # going to allow for training.
    # NOTE: We do not zero out these entries because
    # awkward alters the type in such a way that `track_list_padded.fields` is
    # an empty type (due to a Union type)

    track_list_padded = ak.pad_none(track_list[:, 0:20], 20, axis=1)
    cluster_list_padded = ak.pad_none(cluster_list[:, 0:30], 30, axis=1)
    mseg_list_padded = ak.pad_none(mseg_list[:, 0:30], 30, axis=1)  # type: ignore

    # Next task is to split the padded arrays into their constituent columns.
    def split_array(array: ak.Array, name_prefix: str) -> pd.DataFrame:
        # When splitting, these should be ignored!
        ignore_columns = ["jetIndex"]
        col_elements = len(array[array.fields[0]][0])  # type: ignore

        split_array = ak.Array(
            {
                f"{name_prefix}_{col}_{i_col}": array[col][:, i_col]  # type: ignore
                for i_col in range(0, col_elements)
                for col in array.fields
                if col not in ignore_columns
            }
        )
        return ak.to_dataframe(split_array).reset_index(drop=True)  # type: ignore

    def prefix_array(array: ak.Array, name_prefix: str) -> ak.Array:
        a = ak.Array({f"{name_prefix}_{col}": array[col] for col in array.fields})
        return ak.to_dataframe(a).reset_index(drop=True)  # type: ignore

    split_track_list = split_array(track_list_padded, "track")  # type: ignore
    split_cluster_list = split_array(cluster_list_padded, "clus")  # type: ignore
    split_mseg_list = split_array(mseg_list_padded, "MSeg")  # type: ignore

    df_jet_list = prefix_array(jet_list, "jet")
    df_llps_list = (
        prefix_array(llp_list, "llp") if llp_list is not None else None  # type: ignore
    )
    df_event_list = ak.to_dataframe(event_list).reset_index(drop=True)  # type: ignore

    # Finally combine the arrays  into a single very large DataFrame, and replace all
    # NaN's with 0's.

    df_combine = [
        df_event_list,
        df_jet_list,
        split_track_list,
        split_cluster_list,
        split_mseg_list,
    ]
    if df_llps_list is not None:
        df_combine.append(df_llps_list)

    # Do the combination and some minor clean up.
    df = pd.concat(df_combine, axis=1)
    df_zeroed = df.replace(np.nan, 0.0)
    return df_zeroed 


def signal_processing(
    data: ak.Array, llp_mH: float, llp_mS: float, min_jet_pt: float, max_jet_pt: float
) -> pd.DataFrame:
    """
    Processes the input data to create a pandas DataFrame with relevant columns
    for signal training data.

    Args:
        data (awkward.Array): Input data containing information about jets and LLPs.
        llp_mH (float): Mass of the heavy LLP.
        llp_mS (float): Mass of the light LLP.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns for LLP masses, jet information,
        and a label column.
    """
    # Only look at "good" jets.

    jets_masked = jets_masking(data, min_jet_pt, max_jet_pt)

    # Get the LLP's that are "interesting" for us:
    llp_info = applying_llp_cuts(data.llps)  # type: ignore

    # Now, discard any events that have nothing interesting.
    good_event_mask = (ak.num(llp_info, axis=1) > 0) & (  # type: ignore
        ak.num(jets_masked, axis=1) > 0
    )  # type: ignore
    good_data = data[good_event_mask]
    good_jets_masked = jets_masked[good_event_mask]
    good_llp_info = llp_info[good_event_mask]

    # Find the closest jet index to each LLP, return a list per event.
    # TODO: turn this into dr2
    matches, metric = nearest(
        good_llp_info, good_jets_masked, axis=1, return_metric=True  # type: ignore
    )
    close_matches = metric <= 0.4  # type: ignore

    matched_jets = matches[close_matches]  # type: ignore
    matched_llps = good_llp_info[close_matches]

    # Rebuild the awkward array with these LLP's and jets.
    rebuilt_data = remake_by_replacing(
        good_data, jets=matched_jets, llps=matched_llps  # type: ignore
    )

    # build the pandas df:
    big_df = column_guillotine(rebuilt_data)

    # adding in mH and mS columns
    big_df.insert(0, "llp_mH", float(llp_mH))
    big_df.insert(0, "llp_mS", float(llp_mS))

    # creating the label column, filled with 0s because we're working with signal
    big_df.insert(0, "label", EventType.signal.value)

    # changing the mcEVentWeight to be all 1, matching what Felix does
    big_df["mcEventWeight"] = 1
    return big_df


def bib_processing(
    data: ak.Array, min_jet_pt: float, max_jet_pt: float
) -> Optional[pd.DataFrame]:
    """
    Process data to match BIB HLT jets with actual jets and create a pandas DataFrame.

    Args:
        data (ak.Array): An Awkward Array containing event data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing processed data. Null if nothing was
        found.
    """
    # Make sure we are only working with events with a BIB in them and
    # that we have good jets.
    bib_events_mask = (
        ak.num(data.hlt_jets[data.hlt_jets.isBIB == 1].pt, axis=-1) > 0  # type: ignore
    )
    all_jets_masked = jets_masking(data, min_jet_pt, max_jet_pt)  # type: ignore
    good_event_mask = bib_events_mask & (ak.num(all_jets_masked, axis=1) > 0)  # type: ignore

    data_with_bib = data[good_event_mask]
    jets_masked = all_jets_masked[good_event_mask]

    if len(data_with_bib) == 0:  # type: ignore
        return None

    # and we want to match the BIB HLT jets with the actual jets.
    bib_jets = data_with_bib.hlt_jets[data_with_bib.hlt_jets.isBIB == 1]  # type: ignore
    matched_bib_jets = nearest(bib_jets, jets_masked, axis=1)  # type: ignore

    rebuilt_data = remake_by_replacing(
        data_with_bib, jets=matched_bib_jets, hlt_jets=None  # type: ignore
    )

    big_df = column_guillotine(rebuilt_data)

    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", EventType.BIB.value)
    big_df["mcEventWeight"] = 1

    return big_df


def qcd_processing(
    qcd_data: ak.Array, min_jet_pt: float, max_jet_pt: float
) -> pd.DataFrame:
    """
    Process the given QCD data to prepare it for training a model.

    Args:
        qcd_data (ak.Array): The input QCD data.

    Returns:
        pd.DataFrame: The processed data, with extra columns added and the
        jets sorted by pt.
    """
    # Get a list of all the jets we will consider.
    good_jets = jets_masking(qcd_data, min_jet_pt, max_jet_pt)

    # Next, for the jets we want to consider, sort everything
    # by pt.
    rebuilt = remake_by_replacing(qcd_data, jets=good_jets)  # type: ignore

    # Turn it into a dataframe.
    big_df = column_guillotine(rebuilt)

    # Add the extra columns in.
    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", EventType.QCD.value)

    return big_df


def remake_by_replacing(data: ak.Array, **kwargs: Dict[str, ak.Array]) -> ak.Array:
    """
    Replaces the jets, clusters, tracks, and msegs arrays in the input data with new
    arrays if they are provided. Returns a new ak.Array object with the updated arrays.

    Arguments are the `field` names in `data`.

    If no argument is given, the original arrays in `data` are used.
    If the argument is given as `None` then the original array is dropped.
    If the argument is given as anything else, it is used in the new data.

    Parameters
    ----------
    data : ak.Array
        The input ak.Array object to be updated.
    arg: ak.Array
        The replacement for the contents in `data`

    Returns
    -------
    ak.Array
        A new ak.Array object with the updated arrays. If `jets` is not provided, the
        updated arrays are applied to all events. If `jets` is provided, the updated
        arrays are applied only to events with at least one jet.
    """
    # Build a complete default.
    replacement = {c: data[c] for c in data.fields}

    # Now replace the ones we want to replace from the arguments:
    white_list = ["event", "jets", "clusters", "tracks", "msegs", "llps", "hlt_jets"]
    for k, v in kwargs.items():
        if k not in white_list:
            raise ValueError(
                f"Unknown key {k} in `remake_by_replacing`: use {white_list}"
            )
        if v is None:
            if k in replacement:
                del replacement[k]
        else:
            replacement[k] = v

    new_data = ak.Array(replacement)

    if "jets" not in kwargs.keys():
        return new_data

    return new_data[ak.num(new_data.jets.pt, axis=-1) > 0]  # type: ignore

def load_divert_file(
    file_path: Path, branches: List[str], rename_branches: Optional[Dict[str, str]]
) -> Optional[ak.Array]:
    """
    Load a ROOT file containing DV data and convert it to an awkward array.

    Args:
        file_path (Path): The path to the ROOT file.
        branches (List[str]): A list of branches to load from the file.
        rename_branches (Optional[Dict[str, str]]): A dictionary mapping original
        branch names to new names.

    Returns:
        Optional[ak.Array]: An awkward array containing the loaded data, or None if the
        file has 0 events.
    """
    logging.debug(f"Loading file {file_path}")
    
    with uproot.open(file_path) as in_file:  # type: ignore
        # Check that we don't have an empty file.
        if len(tree) == 0:
            logging.warning(f"File {file_path} has 0 events. Skipped.")
            return None
        data = tree.arrays(branches)  # type: ignore
        # Rename any branches needed
        if rename_branches is not None:
            logging.debug(f"Renaming branches: {rename_branches.keys()}")
            for b_orig, b_new in rename_branches.items():
                if b_orig in data.fields:
                    data[b_new] = data[b_orig]
                    del data[b_orig]

        # Now build the event.
        def zip_common_columns(
            name_stem: str,
            with_name: Optional[str] = "Momentum3D",
            index_column: Optional[str] = None,
        ):
            # add a index column if we've been asked to, and it isn't
            # already there.
            if index_column is not None:
                col_name = name_stem + index_column
                if col_name not in data.fields:
                    index = ak.local_index(data[name_stem + "pt"], axis=1)
                    data[col_name] = index

            stem_len = len(name_stem)
            return ak.zip(
                {c[stem_len:]: data[c] for c in data.fields if c.startswith(name_stem)},
                depth_limit=2,
                with_name=with_name,
            )

        logging.debug("Building the physics objects")
        jets = zip_common_columns("jet_", index_column="jetIndex")
        clusters = zip_common_columns("clus_")
        tracks = zip_common_columns("track_")
        msegs = zip_common_columns("MSeg_", with_name=None)
        llps = zip_common_columns("llp_") if "llp_pt" in data.fields else None
        hlt_jets = (
            zip_common_columns("HLT_jet_") if "HLT_jet_pt" in data.fields else None
        )

        # Make sure jets are sorted. We will sort everything else later on
        # when we've eliminated potentially a lot of events we don't care
        # about.
        logging.debug("Sorting the jets")
        sorted_jet_index = ak.argsort(jets.pt, axis=1, ascending=False)  # type: ignore
        sorted_jets = jets[sorted_jet_index]  # type: ignore

        # And build an array of all the columns that aren't part of anything.
        logging.debug("Building the event level info")
        event_level_info = ak.zip(
            {
                c: data[c]
                for c in data.fields
                if (not c.startswith("jet_"))
                and (not c.startswith("clus_"))
                and (not c.startswith("track_"))
                and (not c.startswith("MSeg_"))
                and (not c.startswith("llp_"))
                and (not c.startswith("HLT_jet_"))
            },
        )
        return ak.Array(
            {
                label: content
                for label, content in [
                    ("event", event_level_info),
                    ("jets", sorted_jets),
                    ("clusters", clusters),
                    ("tracks", tracks),
                    ("msegs", msegs),
                    ("llps", llps),
                    ("hlt_jets", hlt_jets),
                ]
                if content is not None
            }
        )


def convert_divert(config: ConvertDiVertAnalysisConfig):
    """
    Convert DiVert analysis files to pickle format.

    Args:
        config (ConvertDiVertAnalysisConfig): Configuration object containing input and
        output file paths,
            data type information, and other processing parameters.

    Raises:
        ValueError: If no file matching the specified input file pattern is found.

    Returns:
        None
    """
    assert config.input_files is not None, "Must specify an input file for conversion"
    assert (
        config.min_jet_pt is not None and config.max_jet_pt is not None
    ), "Must specify min and max jet pt to convert."
    for f_info in config.input_files:
        found_file = False

        # Build the output file directory
        assert config.output_path is not None
        output_dir_path = config.output_path

        if f_info.output_dir is not None:
            output_dir_path = output_dir_path / f_info.output_dir

        for file_path in (Path(f) for f in glob.glob(str(f_info.input_file))):
            assert file_path.exists(), f"File {file_path} does not exist."
            found_file = True
            logging.info(
                f"Converting files {file_path.name} as a {f_info.data_type} file."
            )

            # The output file is with pkl on it, and in the output directory.
            assert config.output_path is not None
            output_file = output_dir_path / file_path.with_suffix(".pkl").name
            output_parquet = Path('parquet') / output_dir_path / file_path.with_suffix(".parquet").name
            output_parquet.parent.mkdir(parents=True, exist_ok=True)

            if output_file.exists():
                logging.info(f"File {output_file} already exists. Skipping.")
                continue

            # Make sure no one else is working on this file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(output_file) as lock:
                if not lock.is_locked:
                    logging.warning(
                        f"File {output_file} already being processed. Skipping."
                    )
                    continue
                
                # Now run the requested processing
                try:
                    # Check if the file is a parquet file:
                    if os.path.splitext(file_path.name)[1] == '.parquet':
                        data = ak.from_parquet(file_path)
                    
                    # Load up the trees with the proper branches.
                    # Assumed to be a root file
                    else:
                        branches = (
                            config.signal_branches
                            if f_info.data_type == "sig"
                            else (
                                config.qcd_branches
                                if f_info.data_type == "qcd"
                                else config.bib_branches
                            )
                        )
                        assert branches is not None
                        # Saving array as a parquet file for future work
                        
                        ak.to_parquet(data, output_parquet)
                    if data is None:
                        continue
                    

                    # Create output directory
                    output_dir_path.mkdir(parents=True, exist_ok=True)

                    # Process according to the data type.
                    if f_info.data_type == "sig":
                        assert (
                            f_info.llp_mH is not None
                        ), "llp_mH must be set for signal"
                        assert (
                            f_info.llp_mS is not None
                        ), "llp_mS must be set for signal"
                        result = signal_processing(
                            data,  # type: ignore
                            f_info.llp_mH,
                            f_info.llp_mS,
                            config.min_jet_pt,
                            config.max_jet_pt,
                        )
                    elif f_info.data_type == "qcd":
                        result = qcd_processing(
                            data, config.min_jet_pt, config.max_jet_pt  # type: ignore
                        )
                    elif f_info.data_type == "bib":
                        result = bib_processing(
                            data, config.min_jet_pt, config.max_jet_pt  # type: ignore
                        )
                    else:
                        logging.debug("this is bad")
                        raise ValueError(f"Unknown data type {f_info.data_type}")

                    if result is None:
                        logging.warning(
                            f"File {file_path} has 0 events after cuts. Skipped."
                        )
                        continue

                    result.to_pickle(output_file)

                except uproot.exceptions.KeyInFileError as e:  # type:ignore
                    logging.warning(
                        f"File {file_path} does not contain the required branches: "
                        f"{str(e)}. Skipped."
                    )
                    continue
                except Exception as e:
                    logging.error(
                        f"Error processing file {file_path}: {str(e)}. Skipped."
                    )
                    continue

        if not found_file:
            raise ValueError(f"Could not find file matching {f_info.input_file}")
