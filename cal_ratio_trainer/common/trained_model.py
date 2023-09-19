from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from keras import Model
from keras.models import load_model


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


def load_trained_model(run_dir: Path, epoch: int) -> TrainedModel:
    # Make sure the epoch is there.
    assert run_dir / "keras" / f"final_model_weights_{epoch}.h5"

    # Load the model that was written out
    model = load_model(run_dir / "keras" / "final_model.keras")  # type: Optional[Model]
    assert model is not None, "Failed to load model"
    model.load_weights(run_dir / "keras" / f"final_model_weights_{epoch}.keras")

    return TrainedModel(model=model)
