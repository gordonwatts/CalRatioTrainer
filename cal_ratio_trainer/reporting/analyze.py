from pathlib import Path
from typing import List

from attr import dataclass
from cal_ratio_trainer.config import AnalyzeConfig, training_spec
from cal_ratio_trainer.utils import HistoryTracker, find_training_result


@dataclass
class TrainingEpoch:
    """A single training epoch"""

    epoch: int
    loss: float
    accuracy: float


def get_best_results(config: training_spec) -> List[TrainingEpoch]:
    # Find the directory for this training result.
    run_dir = find_training_result(config.name, config.run)
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist.")

    # Now load the history tracker
    epoch_h = HistoryTracker(run_dir / "keras" / "history_checkpoint")
    print(epoch_h)
    return []


def analyze_training_runs(cache: Path, config: AnalyzeConfig):
    """Get and dump the most interesting runs from the list of runs we've done.

    Args:
        cache (Path): Where cache data files are located
        config (AnalyzeConfig): Config to steer our work
    """
    assert config.runs_to_analyze is not None
    best_results = [get_best_results(c) for c in config.runs_to_analyze]

    print(best_results)
