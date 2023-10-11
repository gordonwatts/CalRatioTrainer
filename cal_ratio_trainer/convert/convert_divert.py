import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def applying_llp_cuts(arr):
    # creating the LLP eta/Lxy/Lz masks, based off of the ones mention in the internal
    # note
    central_llp_eta_mask = abs(arr.llp_eta) < 1.4
    endcap_llp_eta_mask = abs(arr.llp_eta) > 1.4

    llp_Lxy_mask = (arr.llp_Lxy > 1200) & (arr.llp_Lxy < 4000)

    llp_Lz_mask = (arr.llp_Lz > 3500) & (arr.llp_Lz < 6000)

    llp_mask = (central_llp_eta_mask & llp_Lxy_mask) | (
        endcap_llp_eta_mask & llp_Lz_mask
    )

    # Create all the LLP info that we find "good":
    llp_info = ak.Array(
        {col: arr[col][llp_mask] for col in arr.fields if col.startswith("llp")}
    )

    return llp_info


def delta_R2(jets, item_eta, item_phi) -> Tuple[ak.Array, ak.Array]:
    "Calculate the DeltaR^2 between each jet and all eta/phi's"

    all_jets = vector.awk(eta=jets.jet_eta, phi=jets.jet_phi, pt=1)
    items = vector.awk(eta=item_eta, phi=item_phi, pt=1)

    # Calculate delta eta, which is, as always, straight forward.
    index = ak.argcartesian({"i_jet": jets.jet_eta, "i_item": item_eta})
    eta_info = ak.cartesian({"j_eta": jets.jet_eta, "eta": item_eta})
    d_eta = eta_info.j_eta - eta_info.eta

    # Calculate delta-phi, which is a bit more complicated.
    # First, we need to make sure that the phi's are in the right range.
    phi_info = ak.cartesian({"j_phi": jets.jet_phi, "phi": item_phi})
    d_phi = np.abs(phi_info.j_phi - phi_info.phi)
    while True:
        mask = d_phi > np.pi
        if not np.any(mask):
            break
        d_phi = (2 * np.pi - d_phi) * mask + d_phi * ~mask

    # Finally we can calculate the delta-R^2.
    delta_r2 = d_eta**2 + d_phi**2
    return delta_r2, index

    # dphi = np.abs(jets.jet_phi - phi)
    # pimask = dphi > np.pi
    # dphi = ((2 * np.pi - dphi) * pimask) + (dphi * ~pimask)
    # deta = jets.jet_eta - eta
    # return np.sqrt(dphi**2 + deta**2)


def closest_jet_index(jets, eta, phi, max_DR: float):
    # Return the `jet_index` of the closest jets in each event to the eta/phi given
    # with an upper bound of max_DR.
    dR = delta_R2(jets, eta, phi)

    pass


def apply_dR_mask(arr, branches, event_type):
    """
    Applying the dR mask onto the array
    dR < 0.4
    event type should be either signal or BIB
    Additionally, removes MSeg/track/clus values that correspond to jets that are not
    of interest
    """
    if event_type == "BIB":
        phi = arr.HLT_jet_phi
        eta = arr.HLT_jet_eta
    else:
        phi = arr.llp_phi
        eta = arr.llp_eta
    dR = delta_R(arr, phi, eta)
    # finding the minimum dR values
    min_index = ak.argmin(dR, axis=1, keepdims=True)
    # filtering to only keep the jet with the smallest dR
    only_min = dR[min_index]
    # 0.4 mask - only want to keep dR < 0.4
    dRmask = only_min < 0.4
    #

    # applying the LLP cuts only if its a signal event
    # mindex = min-index
    if event_type == "signal":
        llp_mask = applying_llp_cuts(arr, branches)
        mindex_mask = ak.flatten(dRmask) & llp_mask
        # applying both the dR mask and the LLP cuts
        dR_masked = ak.Array(
            {
                col: ak.flatten(arr[col][min_index])[mindex_mask]  # type: ignore
                if col.startswith("jet")
                else arr[col][mindex_mask]
                for col in branches
            }
        )
    else:
        # applying only the dR mask - no LLP cuts for BIB
        dR_masked = ak.Array(
            {
                col: ak.flatten(arr[col][min_index])[ak.flatten(dRmask)]  # type: ignore
                if col.startswith("jet")
                else arr[col][ak.flatten(dRmask)]
                for col in branches
            }
        )
        # mindex_mask = ak.flatten(dRmask)

    # Return the list of good jets
    return dR_masked
    # correct_clus, correct_track, correct_mseg = apply_good_jet_mask(
    #     dR_masked, mindex_mask, min_index
    # )
    # return ak.Array(
    #     {
    #         col: dR_masked[col][correct_mseg]
    #         if col.startswith("MSeg")
    #         else dR_masked[col][correct_track]
    #         if col.startswith("track")
    #         else dR_masked[col][correct_clus]
    #         if col.startswith("clus")
    #         else dR_masked[col]
    #         for col in branches
    #     }
    # )


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

    mseg_cols = [
        # TODO: What is the difference between etaPos and etaDir?
        "MSeg_etaPos",
        "MSeg_phiPos",
        "MSeg_etaDir",
        "MSeg_phiDir",
        "MSeg_t0",
        "MSeg_chiSquared",
    ]
    clus_cols = [
        "clus_pt",
        "clus_eta",
        "clus_phi",
        "clus_time",
        "clus_l1ecal",
        "clus_l1hcal",
        "clus_l2ecal",
        "clus_l2hcal",
        "clus_l3ecal",
        "clus_l3hcal",
        "clus_l4ecal",
        "clus_l4hcal",
    ]
    track_cols = [
        "track_pT",
        "track_eta",
        "track_phi",
        "track_chiSquared",
        "track_d0",
        "track_PixelHits",
        "track_SCTHits",
        "track_SCTHoles",
        "track_SCTShared",
        "track_vertex_nParticles",
        "track_z0",
    ]

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
        mask = pairs.jet == pairs.to_match.jetIndex
        if matched_object_nested_jet_index:
            mask = ak.any(mask, axis=-1)

        matched_objects = pairs.to_match[mask]
        return matched_objects

    matched_clusters = expand_and_jet_filter(
        data.jets.jetIndex, data.clusters, matched_object_nested_jet_index=False
    )
    matched_tracks = expand_and_jet_filter(
        data.jets.jetIndex, data.tracks, matched_object_nested_jet_index=True
    )
    matched_msegs = expand_and_jet_filter(
        data.jets.jetIndex, data.msegs, matched_object_nested_jet_index=True
    )

    # Now we have all the bits. Switch to a per-jet view rather than a per-event view.
    jet_list = ak.flatten(data.jets)
    cluster_list = ak.flatten(matched_clusters)
    track_list = ak.flatten(matched_tracks)
    mseg_list = ak.flatten(matched_msegs)

    # Next, lets pad teh cluster, track, and mseg to the length we are
    # going to allow for training.

    return None

    # creating a new array that only contains the

    # padding the array with nones & filling nones with 0s
    # only need to pad out ot the max number of new columns
    # we're going to make

    padded_tcm = ak.Array(
        {
            col: ak.fill_none(ak.pad_none(arr[col], 20, axis=1), 0)
            if col.startswith("track")
            else ak.fill_none(ak.pad_none(arr[col], 30, axis=1), 0)
            if col.startswith("clus")
            else ak.fill_none(ak.pad_none(arr[col], 30, axis=1), 0)
            if col.startswith("MSeg")
            else arr[col]
            for col in branches
        }
    )

    # Creating a new array without any of the track/cluster/mseg columns
    no_tcm = ak.Array(
        {
            col: padded_tcm[col]
            for col in branches
            if not (
                col.startswith("track")
                or col.startswith("clus")
                or col.startswith("MSeg")
            )
        }
    )

    # creating new arrays of *only* the split columns
    mseg_split = ak.Array(
        {
            mseg_name + "_" + str(i_col): padded_tcm[mseg_name][:, i_col]
            for i_col in range(0, 30)
            for mseg_name in mseg_cols
        }
    )
    clus_split = ak.Array(
        {
            clus_name + "_" + str(i_col): padded_tcm[clus_name][:, i_col]
            for i_col in range(0, 30)
            for clus_name in clus_cols
        }
    )
    track_split = ak.Array(
        {
            track_name + "_" + str(i_col): padded_tcm[track_name][:, i_col]
            for i_col in range(0, 20)
            for track_name in track_cols
        }
    )

    no_tcm_df = (ak.to_dataframe(no_tcm)).reset_index(drop=True)  # type: ignore
    mseg_split_df = (ak.to_dataframe(mseg_split)).reset_index(drop=True)  # type: ignore
    clus_split_df = (ak.to_dataframe(clus_split)).reset_index(drop=True)  # type: ignore
    track_split_df = (ak.to_dataframe(track_split)).reset_index(  # type: ignore
        drop=True
    )

    df_combine = [no_tcm_df, mseg_split_df, clus_split_df, track_split_df]

    return pd.concat(df_combine, axis=1)


def signal_processing(signal_file_branch, llp_mH: float, llp_mS: float) -> pd.DataFrame:
    # Only look at "good" jets.
    jet_masked = jets_masking(signal_file_branch, signal_file_branch.fields)

    # Get the LLP's that are "interesting" for us:
    llp_info = applying_llp_cuts(jet_masked)

    # Find the closest jet index to each LLP, return a list per event.
    matched_jet_index = closest_jet_index(
        jet_masked, llp_info.llp_eta, llp_info.llp_phi, max_DR=0.4
    )
    # splitting up into the 0th and 1th LLPs

    jet_masked_0 = ak.Array(
        {
            col: jet_masked[col][:, 0] if col.startswith("llp") else jet_masked[col]
            for col in signal_file_branch.fields
        }
    )
    jet_masked_1 = ak.Array(
        {
            col: jet_masked[col][:, 1] if col.startswith("llp") else jet_masked[col]
            for col in signal_file_branch.fields
        }
    )

    dR_masked_0 = apply_dR_mask(jet_masked_0, signal_file_branch.fields, "signal")
    dR_masked_1 = apply_dR_mask(jet_masked_1, signal_file_branch.fields, "signal")

    sorted_tcm_0 = sorting_by_pT(dR_masked_0, signal_file_branch.fields)
    sorted_tcm_1 = sorting_by_pT(dR_masked_1, signal_file_branch.fields)
    big_df = pd.concat(
        [
            column_guillotine(sorted_tcm_0, signal_file_branch.fields),
            column_guillotine(sorted_tcm_1, signal_file_branch.fields),
        ],
        axis=0,
    )

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

    new_data = ak.Array(
        {
            "jets": new_jets,
            "clusters": new_clusters,
            "tracks": new_tracks,
            "msegs": new_msegs,
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


# vector.backends.awkward.MomentumAwkward3D
class DiVertJetArray(ak.Array):
    def clusters(self, clusters: ak.Array) -> ak.Array:
        return ak.Array({"pt": [1, 2, 3], "eta": {4, 5, 6}})


ak.behavior["*", "DiVertJetArray"] = DiVertJetArray


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

        # # if there is no next array on the event, then we had better create one.
        # if "jetIndex" not in jets.fields:  # type: ignore
        #     # TODO: Integrate this with `zip_common_columns` so we run ak.zip only once.
        #     index = ak.local_index(jets.pt, axis=1)  # type: ignore
        #     # index = ak.argsort(jets.pt, axis=1, ascending=False)  # type: ignore
        #     # sorted_index = ak.sort(index, axis=1)  # type: ignore
        #     jets = ak.zip(
        #         {"jetIndex": index, **{c: jets[c] for c in jets.fields}},  # type:ignore
        #         with_name="Momentum3D",
        #     )

        # Make sure jets are sorted. We will sort everything else later on
        # when we've eliminated potentially a lot of events we don't care
        # about.
        sorted_jet_index = ak.argsort(jets.pt, axis=1, ascending=False)  # type: ignore
        sorted_jets = jets[sorted_jet_index]  # type: ignore

        return ak.Array(
            {
                "jets": sorted_jets,
                "clusters": clusters,
                "tracks": tracks,
                "msegs": msegs,
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

                    # with uproot.open(file_path) as in_file:  # type: ignore
                    #     # Check that we don't have an empty file.
                    #     tree = in_file["trees_DV_"]
                    #     if len(tree) == 0:
                    #         logging.warning(f"File {file_path} has 0 events. Skipped.")
                    #         continue

                    #     assert branches is not None
                    #     extra_branches = (
                    #         [
                    #             "HLT_jet_isBIB",
                    #             "HLT_jet_phi",
                    #             "HLT_jet_eta",
                    #             "nn_jet_index",
                    #         ]
                    #         if f_info.data_type == "bib"
                    #         else ["nn_jet_index"]
                    #     )
                    #     data = tree.arrays(branches + extra_branches)  # type: ignore
                    #     if config.rename_branches is not None:
                    #         for b_orig, b_new in config.rename_branches.items():
                    #             if b_orig in data.fields:
                    #                 data[b_new] = data[b_orig]
                    #                 del data[b_orig]
                    #                 if b_orig in extra_branches:
                    #                     extra_branches.remove(b_orig)
                    #                     extra_branches.append(b_new)

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
                    if len(extra_branches) > 0:
                        result = result.drop(columns=extra_branches)
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
