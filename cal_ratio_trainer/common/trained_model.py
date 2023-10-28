import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from keras import Model
from keras.models import load_model as load_keras_model
from keras.models import model_from_json as load_keras_model_from_json

from cal_ratio_trainer.utils import find_training_result


@dataclass
class TrainedModelData:
    "Information about an epoch along with test data we can use"

    # The input data for the model
    x: List[np.ndarray]

    # The expected results (labels) for the test data
    y: pd.Series

    # The LLP information (mH, mS, etc.) for the training data
    z: pd.DataFrame

    # Weights associated with the x, y, and z.
    weights: List[np.ndarray]


@dataclass
class TrainedModel:
    # The loaded model and weights
    model: Model

    def predict(self, x: TrainedModelData) -> np.ndarray:
        return self.model.predict(x.x, verbose="0")


def _load_model_only(run_dir: Path) -> Model:
    # Load the model that was written out
    model = load_keras_model(run_dir / "keras" / "final_model.keras")
    assert model is not None, "Failed to load model"
    return model


def load_trained_model_from_training(run_dir: Path, epoch: int) -> TrainedModel:
    """
    Load a trained model from a given training run directory and epoch.

    Args:
        run_dir (Path): The directory containing the training run.
        epoch (int): The epoch number of the trained model to load.

    Returns:
        TrainedModel: An instance of the TrainedModel class containing the
                loaded model.
    """
    # Make sure the epoch is there.
    epoch_path = run_dir / "keras" / f"final_model_weights_{epoch}.keras"
    assert epoch_path.exists(), f"Trained Model Keras file Not Found: {epoch_path}"

    # Load the model that was written out
    model = _load_model_only(run_dir)
    model.load_weights(epoch_path)

    return TrainedModel(model=model)


def load_trained_model_from_json(json_path: Path) -> Model:
    # Load the model that was written out
    assert json_path.exists(), f"Trained Model JSON file Not Found: {json_path}"

    with json_path.open() as f:
        fdeep_model_info = json.load(f)

    model = load_keras_model_from_json(json.dumps(fdeep_model_info["architecture"]))
    assert model is not None, f"Failed to load model {json_path}"

    return model


def load_model(info: str, base_path: Path = Path(".")) -> Model:
    # determine if this is a path or a training string.
    try:
        p = Path(info)
        if p.exists():
            # This should be a JSON file!
            return load_trained_model_from_json(p)
    except Exception:
        pass

    # Ok - if here, then we have an actual training.
    name, run = info.split("/")
    model_dir = find_training_result(name, int(run), base_path)
    return _load_model_only(model_dir)
