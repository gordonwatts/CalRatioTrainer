from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from keras import Model
from keras.models import load_model, model_from_json


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
    model = load_model(run_dir / "keras" / "final_model.keras")  # type: Optional[Model]
    assert model is not None, "Failed to load model"
    model.load_weights(epoch_path)

    return TrainedModel(model=model)


def load_trained_model_from_json(json_path: Path) -> TrainedModel:
    # Load the model that was written out
    assert json_path.exists(), f"Trained Model JSON file Not Found: {json_path}"

    model = model_from_json(json_path.read_text())
    assert model is not None, f"Failed to load model {json_path}"

    return TrainedModel(model=model)
