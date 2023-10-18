import glob
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import pandas as pd
import uproot
import vector

from cal_ratio_trainer.common.file_lock import FileLock
from cal_ratio_trainer.config import ConvertDiVertAnalysisConfig

# Much of the code was copied directly from Alex Golub's code on gitlab.
# Many thanks to their work fro this!

vector.register_awkward()


def jets_masking(array: ak.Array) -> ak.Array:
    """
    Returns an awkward array containing only good, central jets with pT between 40 and 500 GeV.

    Parameters:
    -----------
    array : ak.Array
        An awkward array containing jet information.

    Returns:
    --------
    ak.Array
        An awkward array containing only good, central jets with pT between 40 and 500 GeV.
    """
    # Only good, central, jets are considered.
    jet_pT_mask = (array.jets.pt >= 40) & (array.jets.pt < 500)  # type: ignore
    jet_eta_mask = np.abs(array.jets.eta) <= 2.5  # type: ignore

    return array.jets[jet_pT_mask & jet_eta_mask]  # type: ignore


# def apply_good_jet_mask(arr, mindex_mask, min_index):
#     """
#     Creating the good jet indices mask and then searching through the clus/track/mseg
#     jetIndex values to find which ones match
#     This function serves to not need to do 'as much' copy/paste
#     """
#     good_jet_indices = ak.flatten(min_index[mindex_mask])

#     correct_clusters = arr.cluster_jetIndex == good_jet_indices
#     correct_track = ak.flatten(arr.track_jetIndex, axis=-1) == good_jet_indices
#     correct_mseg = mseg_filter(arr.MSeg_jetIndex, good_jet_indices)

#     return (correct_clusters, correct_track, correct_mseg)


def mseg_filter(mseg, correct_jets):
    """
    function to return a mseg filter
    creates a double nested array called output that contains information about whether
    the mseg jet index matches with the 'good' jet index we've established
    """
    output = []
    for i, x in enumerate(mseg):
        nested_output = []
        for j in x:
            nested_output.append(correct_jets[i] in j)
        output.append(ak.Array(nested_output))
    return ak.Array(output)


def applying_llp_cuts(llps):
    # creating the LLP eta/Lxy/Lz masks, based off of the ones mention in the internal
    # note
    central_llp_eta_mask = abs(llps.eta) < 1.4
    endcap_llp_eta_mask = abs(llps.eta) > 1.4

    llp_Lxy_mask = (llps.Lxy > 1200) & (llps.Lxy < 4000)

    llp_Lz_mask = (llps.Lz > 3500) & (llps.Lz < 6000)

    llp_mask = (central_llp_eta_mask & llp_Lxy_mask) | (
        endcap_llp_eta_mask & llp_Lz_mask
    )

    # Create all the LLP info that we find "good":
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
    mval, (a, b) = metric_table(first, other, axis, metric, return_combinations=True)
    if axis is None:
        # NotImplementedError: awkward.firsts with axis=-1
        axis = other.layout.purelist_depth - 2
    assert axis is not None
    mmin = ak.argmin(mval, axis=axis + 1, keepdims=True)
    out = ak.firsts(b[mmin], axis=axis + 1)  # type: ignore
    metric = ak.firsts(mval[mmin], axis=axis + 1)  # type: ignore
    if threshold is not None:
        out = out.mask[metric <= threshold]  # type: ignore
    if return_metric:
        return out, metric
    return out


# def apply_dR_mask(arr, branches, event_type):
#     """
#     Applying the dR mask onto the array
#     dR < 0.4
#     event type should be either signal or BIB
#     Additionally, removes MSeg/track/clus values that correspond to jets that are not
#     of interest
#     """
#     if event_type == "BIB":
#         phi = arr.HLT_jet_phi
#         eta = arr.HLT_jet_eta
#     else:
#         phi = arr.llp_phi
#         eta = arr.llp_eta
#     dR = delta_R(arr, phi, eta)
#     # finding the minimum dR values
#     min_index = ak.argmin(dR, axis=1, keepdims=True)
#     # filtering to only keep the jet with the smallest dR
#     only_min = dR[min_index]
#     # 0.4 mask - only want to keep dR < 0.4
#     dRmask = only_min < 0.4
#     #

#     # applying the LLP cuts only if its a signal event
#     # mindex = min-index
#     if event_type == "signal":
#         llp_mask = applying_llp_cuts(arr, branches)
#         mindex_mask = ak.flatten(dRmask) & llp_mask
#         # applying both the dR mask and the LLP cuts
#         dR_masked = ak.Array(
#             {
#                 col: ak.flatten(arr[col][min_index])[mindex_mask]  # type: ignore
#                 if col.startswith("jet")
#                 else arr[col][mindex_mask]
#                 for col in branches
#             }
#         )
#     else:
#         # applying only the dR mask - no LLP cuts for BIB
#         dR_masked = ak.Array(
#             {
#                 col: ak.flatten(arr[col][min_index])[ak.flatten(dRmask)]  # type: ignore
#                 if col.startswith("jet")
#                 else arr[col][ak.flatten(dRmask)]
#                 for col in branches
#             }
#         )
#         # mindex_mask = ak.flatten(dRmask)

#     # Return the list of good jets
#     return dR_masked
#     # correct_clus, correct_track, correct_mseg = apply_good_jet_mask(
#     #     dR_masked, mindex_mask, min_index
#     # )
#     # return ak.Array(
#     #     {
#     #         col: dR_masked[col][correct_mseg]
#     #         if col.startswith("MSeg")
#     #         else dR_masked[col][correct_track]
#     #         if col.startswith("track")
#     #         else dR_masked[col][correct_clus]
#     #         if col.startswith("clus")
#     #         else dR_masked[col]
#     #         for col in branches
#     #     }
#     # )


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
    new_clusters = data.clusters[new_clusters_index]

    new_track_index = ak.argsort(
        data.tracks.pt,  # type: ignore
        axis=1,
        ascending=False,
    )
    new_tracks = data.tracks[new_track_index]

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
    This array should be sorted already

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
        data.jets.jetIndex,  # type: ignore
        data.clusters,  # type: ignore
        matched_object_nested_jet_index=False,
    )
    matched_tracks = expand_and_jet_filter(
        data.jets.jetIndex,  # type: ignore
        data.tracks,  # type: ignore
        matched_object_nested_jet_index=True,
    )
    matched_msegs = expand_and_jet_filter(
        data.jets.jetIndex,  # type: ignore
        data.msegs,  # type: ignore
        matched_object_nested_jet_index=True,
    )

    # Now we have all the bits. Switch to a per-jet view rather than a per-event view.
    jet_list = ak.flatten(data.jets)
    llp_list = None if "llps" not in data.fields else ak.flatten(data.llps)
    cluster_list = ak.flatten(matched_clusters)
    track_list = ak.flatten(matched_tracks)
    mseg_list = ak.flatten(matched_msegs)

    # We also have to deal with the per-event view. Each row has to be
    # duplicated the right number of times.

    event_x_jet = ak.cartesian({"event": data.event, "jet": data.jets})
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

    # Next, lets pad, with zeros, the cluster, track, and mseg to the length we are
    # going to allow for training.
    # NOTE: We do not zero out these entries because
    # awkward alters the type in such a way that `track_list_padded.fields` is
    # an empty type (due to a Union type)

    track_list_padded = ak.pad_none(track_list, 20, axis=1)
    cluster_list_padded = ak.pad_none(cluster_list, 30, axis=1)
    mseg_list_padded = ak.pad_none(mseg_list, 30, axis=1)

    # Next task is to split the padded arrays into their constituent columns.
    def split_array(array: ak.Array, name_prefix: str) -> pd.DataFrame:
        # When splitting, these should be ignored!
        ignore_columns = ["jetIndex"]
        col_elements = len(array[array.fields[0]][0])  # type: ignore

        split_array = ak.Array(
            {
                f"{name_prefix}_{col}_{i_col}": array[col][:, i_col]
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


def signal_processing(data, llp_mH: float, llp_mS: float) -> pd.DataFrame:
    # Only look at "good" jets.
    jets_masked = jets_masking(data)

    # Get the LLP's that are "interesting" for us:
    llp_info = applying_llp_cuts(data.llps)

    # Find the closest jet index to each LLP, return a list per event.
    # TODO: turn this into dr2
    matches, metric = nearest(llp_info, jets_masked, axis=1, return_metric=True)
    close_matches = metric <= 0.4  # type: ignore

    matched_jets = matches[close_matches]  # type: ignore
    matched_llps = llp_info[close_matches]

    # Rebuild the awkward array with these LLP's and jets.
    rebuilt_data = remake_by_replacing(data, jets=matched_jets, llps=matched_llps)

    # build the pandas df:
    big_df = column_guillotine(rebuilt_data)

    # adding in mH and mS columns
    big_df.insert(0, "llp_mH", float(llp_mH))
    big_df.insert(0, "llp_mS", float(llp_mS))

    # creating the label column, filled with 0s because we're working with signal
    big_df.insert(0, "label", int(0))

    # changing the mcEVentWeight to be all 1, matching what Felix does
    big_df["mcEventWeight"] = 1

    return big_df


def bib_processing(bib_data) -> pd.DataFrame:
    # tracking if the HLT jet is BIB
    is_bib_mask = bib_data.HLT_jet_isBIB == 1
    bib_masked = ak.Array(
        {
            col: bib_data[col][is_bib_mask] if col.startswith("HLT") else bib_data[col]
            for col in bib_data.fields
        }
    )
    length_mask = ak.num(bib_masked.HLT_jet_isBIB, axis=-1) > 0  # type: ignore
    bib_masked = ak.Array(
        {col: bib_masked[col][length_mask] for col in bib_data.fields}
    )

    jet_masked = jets_masking(bib_masked, bib_data.fields)

    # keeping only the 1st HLT jet - should be fixed later but fine for now
    jet_masked = ak.Array(
        {
            col: jet_masked[col][:, 0] if col.startswith("HLT") else jet_masked[col]
            for col in bib_data.fields
        }
    )

    dR_masked = apply_dR_mask(jet_masked, bib_data.fields, "BIB")
    sorted_tcm = sorting_by_pT(dR_masked, bib_data.fields)
    big_df = column_guillotine(sorted_tcm, bib_data.fields)

    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", 2)
    big_df["mcEventWeight"] = 1

    return big_df


def remake_by_replacing(
    data: ak.Array,
    jets: Optional[ak.Array] = None,
    clusters: Optional[ak.Array] = None,
    msegs: Optional[ak.Array] = None,
    tracks: Optional[ak.Array] = None,
    llps: Optional[ak.Array] = None,
) -> ak.Array:
    """
    Replaces the jets, clusters, tracks, and msegs arrays in the input data with new
    arrays if they are provided. Returns a new ak.Array object with the updated arrays.

    Parameters
    ----------
    data : ak.Array
        The input ak.Array object to be updated.
    jets : ak.Array, optional
        The new jets array to replace the one in `data`. If not provided, the original
        jets array in `data` is used.
    clusters : ak.Array, optional
        The new clusters array to replace the one in `data`. If not provided, the
        original clusters array in `data` is used.
    msegs : ak.Array, optional
        The new msegs array to replace the one in `data`. If not provided, the original
        msegs array in `data` is used.
    tracks : ak.Array, optional
        The new tracks array to replace the one in `data`. If not provided, the
        original tracks array in `data` is used.
    llps : ak.Array, optional
        The new LLPs' list to replace the one in data.

    Returns
    -------
    ak.Array
        A new ak.Array object with the updated arrays. If `jets` is not provided, the
        updated arrays are applied to all events. If `jets` is provided, the updated
        arrays are applied only to events with at least one jet.
    """
    new_jets = jets if jets is not None else data.jets
    new_clusters = clusters if clusters is not None else data.clusters
    new_tracks = tracks if tracks is not None else data.tracks
    new_msegs = msegs if msegs is not None else data.msegs
    new_llps = (
        llps if llps is not None else data.llps if "llps" in data.fields else None
    )

    new_data = ak.Array(
        {
            "event": data.event,
            "jets": new_jets,
            "clusters": new_clusters,
            "tracks": new_tracks,
            "msegs": new_msegs,
            "llps": new_llps,
        }
    )

    if jets is None:
        return new_data

    return new_data[ak.num(jets.pt, axis=-1) > 0]  # type: ignore


def qcd_processing(qcd_data) -> pd.DataFrame:
    # Get a list of all the jets we will consider.
    good_jets = jets_masking(qcd_data)

    # Next, for the jets we want to consider, sort everything
    # by pt.
    sorted = sort_by_pt(remake_by_replacing(qcd_data, jets=good_jets))

    # Turn it into a dataframe.
    big_df = column_guillotine(sorted)

    # Add the extra columns in.
    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", 1)

    return big_df


def load_divert_file(
    file_path: Path, branches: List[str], rename_branches: Optional[Dict[str, str]]
) -> Optional[ak.Array]:
    with uproot.open(file_path) as in_file:  # type: ignore
        # Check that we don't have an empty file.
        tree = in_file["trees_DV_"]
        if len(tree) == 0:
            logging.warning(f"File {file_path} has 0 events. Skipped.")
            return None

        data = tree.arrays(branches)  # type: ignore

        # Rename any branches needed
        if rename_branches is not None:
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

        jets = zip_common_columns("jet_", index_column="jetIndex")
        clusters = zip_common_columns("clus_")
        tracks = zip_common_columns("track_")
        msegs = zip_common_columns("MSeg_", with_name=None)
        llps = zip_common_columns("llp_") if "llp_pt" in data.fields else None

        # Make sure jets are sorted. We will sort everything else later on
        # when we've eliminated potentially a lot of events we don't care
        # about.
        sorted_jet_index = ak.argsort(jets.pt, axis=1, ascending=False)  # type: ignore
        sorted_jets = jets[sorted_jet_index]  # type: ignore

        # And build an array of all the columns that aren't part of anything.
        event_level_info = ak.zip(
            {
                c: data[c]
                for c in data.fields
                if (not c.startswith("jet_"))
                and (not c.startswith("clus_"))
                and (not c.startswith("track_"))
                and (not c.startswith("MSeg_"))
                and (not c.startswith("llp_"))
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
                ]
                if content is not None
            }
        )


def convert_divert(config: ConvertDiVertAnalysisConfig):
    assert config.input_files is not None

    for f_info in config.input_files:
        found_file = False

        # Build the output file directory
        assert config.output_path is not None
        output_dir_path = config.output_path

        if f_info.output_dir is not None:
            output_dir_path = output_dir_path / f_info.output_dir

        for file_path in (Path(f) for f in glob.glob(str(f_info.input_file))):
            assert file_path.exists()
            found_file = True
            logging.info(
                f"Converting files {file_path.name} as a {f_info.data_type} file."
            )

            # The output file is with pkl on it, and in the output directory.
            assert config.output_path is not None
            output_file = output_dir_path / file_path.with_suffix(".pkl").name

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
                    # Load up the trees with the proper branches.
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

                    data = load_divert_file(file_path, branches, config.rename_branches)

                    # Create output directory
                    output_dir_path.mkdir(parents=True, exist_ok=True)

                    # Process according to the data type.
                    if f_info.data_type == "sig":
                        assert f_info.llp_mH is not None
                        assert f_info.llp_mS is not None
                        result = signal_processing(
                            data,
                            f_info.llp_mH,
                            f_info.llp_mS,
                        )
                    elif f_info.data_type == "qcd":
                        result = qcd_processing(data)
                    elif f_info.data_type == "bib":
                        result = bib_processing(data)
                    else:
                        raise ValueError(f"Unknown data type {f_info.data_type}")

                    # Write the output file
                    # if len(extra_branches) > 0:
                    #     result = result.drop(columns=extra_branches)
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
