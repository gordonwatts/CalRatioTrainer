import glob
import logging
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from cal_ratio_trainer.common.file_lock import FileLock

from cal_ratio_trainer.config import ConvertDiVertAnalysisConfig

# Much of the code was copied directly from Alex Golub's code on gitlab.
# Many thanks to their work fro this!


def jets_masking(array, branches):
    """
    Applies various cuts on the jets
    Returns masked array
    """

    jet_pT_mask = (array.jet_pT >= 40) & (array.jet_pT < 500)
    jet_eta_mask = (array.jet_eta >= -2.5) & (array.jet_eta <= 2.5)
    masked = ak.Array(
        {
            col: array[col][jet_pT_mask & jet_eta_mask]
            if col.startswith("jet")
            else array[col]
            for col in branches
        }
    )
    length_mask = ak.num(masked.jet_pT, axis=-1) > 0  # type: ignore
    # TODO: This could probably be just `masked[length_mask]`.
    return ak.Array({col: masked[col][length_mask] for col in branches})


def apply_good_jet_mask(arr, mindex_mask, min_index):
    """
    Creating the good jet indices mask and then searching through the clus/track/mseg
    jetIndex values to find which ones match
    This function serves to not need to do 'as much' copy/paste
    """
    good_jet_indices = ak.flatten(min_index[mindex_mask])

    correct_clusters = arr.cluster_jetIndex == good_jet_indices
    correct_track = ak.flatten(arr.track_jetIndex, axis=-1) == good_jet_indices
    correct_mseg = mseg_filter(arr.MSeg_jetIndex, good_jet_indices)

    return (correct_clusters, correct_track, correct_mseg)


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


def applying_llp_cuts(arr, branches):
    # creating the LLP eta/Lxy/Lz masks, based off of the ones mention in the internal
    # note
    central_llp_eta_mask = abs(arr.llp_eta) < 1.4
    endcap_llp_eta_mask = abs(arr.llp_eta) > 1.4

    llp_Lxy_mask = (arr.llp_Lxy > 1200) & (arr.llp_Lxy < 4000)

    llp_Lz_mask = (arr.llp_Lz > 3500) & (arr.llp_Lz < 6000)

    return (central_llp_eta_mask & llp_Lxy_mask) | (endcap_llp_eta_mask & llp_Lz_mask)


def delta_R(arr, phi, eta):
    """
    doing a dR calculation for llp or HLT_jet and jet
    phi should be passed in as arr.llp_phi or arr.HLT_jet_phi, etc.
    eta should be arr.llp_eta, arr.HLT_jet_eta
    """
    dphi = np.abs(arr.jet_phi - phi)
    pimask = dphi > np.pi
    dphi = ((2 * np.pi - dphi) * pimask) + (dphi * ~pimask)
    deta = arr.jet_eta - eta
    return np.sqrt(dphi**2 + deta**2)


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
        mindex_mask = ak.flatten(dRmask)
    correct_clus, correct_track, correct_mseg = apply_good_jet_mask(
        dR_masked, mindex_mask, min_index
    )
    return ak.Array(
        {
            col: dR_masked[col][correct_mseg]
            if col.startswith("MSeg")
            else dR_masked[col][correct_track]
            if col.startswith("track")
            else dR_masked[col][correct_clus]
            if col.startswith("clus")
            else dR_masked[col]
            for col in branches
        }
    )


def sorting_by_pT(arr, branches):
    """
    takes in an array, creates a sorting mask based on pt
    and returns the sorted array
    """

    # creating the sorting mask
    clus_sort = ak.argsort(arr.clus_pt, axis=1)
    track_sort = ak.argsort(arr.track_pT, axis=1)

    # applying the sorts to the columns
    return ak.Array(
        {
            col: arr[col][track_sort]
            if col.startswith("track")
            else arr[col][clus_sort]
            if col.startswith("clus")
            else arr[col]
            for col in branches
        }
    )


def column_guillotine(arr, branches):
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


def signal_processing(
    signal_file, llp_mH: float, llp_mS: float, branches: List[str], output_file: Path
):
    # getting the specific branches as defined by branches
    # should be 'trees_DV_' for every file
    signal_file_branch = signal_file["trees_DV_"].arrays(branches)

    jet_masked = jets_masking(signal_file_branch, branches)

    # splitting up into the 0th and 1th LLPs

    jet_masked_0 = ak.Array(
        {
            col: jet_masked[col][:, 0] if col.startswith("llp") else jet_masked[col]
            for col in branches
        }
    )
    jet_masked_1 = ak.Array(
        {
            col: jet_masked[col][:, 1] if col.startswith("llp") else jet_masked[col]
            for col in branches
        }
    )

    dR_masked_0 = apply_dR_mask(jet_masked_0, branches, "signal")
    dR_masked_1 = apply_dR_mask(jet_masked_1, branches, "signal")

    sorted_tcm_0 = sorting_by_pT(dR_masked_0, branches)
    sorted_tcm_1 = sorting_by_pT(dR_masked_1, branches)
    big_df = pd.concat(
        [
            column_guillotine(sorted_tcm_0, branches),
            column_guillotine(sorted_tcm_1, branches),
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

    # writes it out to pickle file
    # this should be removed when the infrastructure for reading in multiple root files
    # is added as the pickle file should be written out after reading in all of them
    big_df.to_pickle(output_file)

    return big_df


def bib_processing(file, base_branches: List[str], output_file: Path):
    # need to add in dR calculation
    extra_branches = ["HLT_jet_isBIB", "HLT_jet_phi", "HLT_jet_eta"]
    branches = base_branches + extra_branches
    bib_data = file["trees_DV_"].arrays(branches)

    # tracking if the HLT jet is BIB
    is_bib_mask = bib_data.HLT_jet_isBIB == 1
    bib_masked = ak.Array(
        {
            col: bib_data[col][is_bib_mask] if col.startswith("HLT") else bib_data[col]
            for col in branches
        }
    )
    length_mask = ak.num(bib_masked.HLT_jet_isBIB, axis=-1) > 0  # type: ignore
    bib_masked = ak.Array({col: bib_masked[col][length_mask] for col in branches})

    jet_masked = jets_masking(bib_masked, branches)

    # keeping only the 1st HLT jet - should be fixed later but fine for now
    jet_masked = ak.Array(
        {
            col: jet_masked[col][:, 0] if col.startswith("HLT") else jet_masked[col]
            for col in branches
        }
    )

    dR_masked = apply_dR_mask(jet_masked, branches, "BIB")
    sorted_tcm = sorting_by_pT(dR_masked, branches)
    big_df = column_guillotine(sorted_tcm, branches)

    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", 2)
    big_df["mcEventWeight"] = 1

    # Remove the extra branches we needed for processing
    big_df = big_df.drop(columns=extra_branches)

    big_df.to_pickle(output_file)

    return big_df


def qcd_processing(file, branches: List[str], output_file: Path):
    qcd_data = file["trees_DV_"].arrays(branches)
    jet_masked = jets_masking(qcd_data, branches)

    sorted = sorting_by_pT(jet_masked, branches)
    big_df = column_guillotine(sorted, branches)

    # Add the extra columns in.
    big_df.insert(0, "llp_Lz", 0.0)
    big_df.insert(0, "llp_Lxy", 0.0)
    big_df.insert(0, "llp_phi", 0.0)
    big_df.insert(0, "llp_eta", 0.0)
    big_df.insert(0, "llp_pT", 0.0)

    big_df.insert(0, "llp_mH", 0.0)
    big_df.insert(0, "llp_mS", 0.0)
    big_df.insert(0, "label", 1)

    big_df.to_pickle(output_file)

    return big_df


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
                    with uproot.open(file_path) as in_file:  # type: ignore
                        # Check that we don't have an empty file.
                        data = in_file["trees_DV_"]
                        if len(data) == 0:
                            logging.warning(f"File {file_path} has 0 events. Skipped.")
                            continue

                        # Create output directory
                        output_dir_path.mkdir(parents=True, exist_ok=True)

                        # Process according to the data type.
                        if f_info.data_type == "sig":
                            assert config.signal_branches is not None
                            assert f_info.llp_mH is not None
                            assert f_info.llp_mS is not None
                            signal_processing(
                                in_file,
                                f_info.llp_mH,
                                f_info.llp_mS,
                                config.signal_branches,
                                output_file,
                            )
                        elif f_info.data_type == "qcd":
                            assert config.qcd_branches is not None
                            qcd_processing(in_file, config.qcd_branches, output_file)
                        elif f_info.data_type == "bib":
                            assert config.bib_branches is not None
                            bib_processing(in_file, config.bib_branches, output_file)
                        else:
                            raise ValueError(f"Unknown data type {f_info.data_type}")
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
