# The cluster column names needed for training.
# NOTE: The ordering is important, as are the numbers (that are hardwired)!
# Then network is built expecting this - so not change them!!!
from typing import List
from enum import Enum

col_cluster_names_raw = [
    "clus_pt",
    "clus_eta",
    "clus_phi",
    "clus_l1hcal",
    "clus_l1ecal",
    "clus_l2hcal",
    "clus_l2ecal",
    "clus_l3hcal",
    "clus_l3ecal",
    "clus_l4ecal",
    "clus_l4hcal",
    "clus_time",
]


def _generate_col_cluster_names(index: int) -> List[str]:
    "Generate the column names by appending a _{index} to each column name"
    return [f"{name}_{index}" for name in col_cluster_names_raw]


col_track_names_raw = [
    "track_pt",
    "track_eta",
    "track_phi",
    "track_vertex_nParticles",
    "track_d0",
    "track_z0",
    "track_chiSquared",
    "track_PixelShared",
    "track_SCTShared",
    "track_PixelHoles",
    "track_SCTHoles",
    "track_PixelHits",
    "track_SCTHits",
]


def _generate_col_track_names(index: int) -> List[str]:
    "Generate the column names by appending a _{index} to each column name"
    return [f"{name}_{index}" for name in col_track_names_raw]


col_mseg_names_raw = [
    "MSeg_etaPos",
    "MSeg_phiPos",
    "MSeg_etaDir",
    "MSeg_phiDir",
    "MSeg_chiSquared",
    "MSeg_t0",
]


def _generate_col_mseg_names(index: int) -> List[str]:
    "Generate the column names by appending a _{index} to each column name"
    return [f"{name}_{index}" for name in col_mseg_names_raw]


# List of all the cluster names from 0 to 29
col_cluster_names = sum([_generate_col_cluster_names(i) for i in range(30)], [])

# List of all the track names from 0 to 19
col_track_names = sum([_generate_col_track_names(i) for i in range(20)], [])

# List of all mseg names from 0 to 29
col_mseg_names = sum([_generate_col_mseg_names(i) for i in range(30)], [])

# list of cluster, track, and mseg names
col_cluster_track_mseg_names = col_cluster_names + col_track_names + col_mseg_names

col_jet_names = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
]


col_llp_mass_names = ["llp_mH", "llp_mS"]

event_level_names = ["label", "eventNumber", "mcEventWeight", "runNumber"]

col_llp_names = ["llp_eta", "llp_phi", "llp_Lxy", "llp_Lz", "llp_pt"]
all_cols = (
    event_level_names
    + col_llp_mass_names
    + col_jet_names
    + col_cluster_track_mseg_names
    + col_llp_names
)


class EventType(Enum):
    "The magic numbers for data labeling"
    QCD = 0
    signal = 1
    BIB = 2
