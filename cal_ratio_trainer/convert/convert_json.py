import os
from pathlib import Path
from cal_ratio_trainer.common.trained_model import (
    load_model_from_spec,
)
from cal_ratio_trainer.config import ConvertTrainingConfig


def convert_file(c: ConvertTrainingConfig):
    # Load up the model and weights for the run.
    assert c.run_to_convert is not None
    model = load_model_from_spec(c.run_to_convert)

    # Now write it out!
    assert c.output_json is not None
    output_path = Path(c.output_json)

    # Make sure the directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write out as keras output
    model.model.save(output_path.with_suffix(".keras"))

    # Next, convert it to f-deep format.
    convert_file = Path(__file__).parent / "fdeep" / "keras_export" / "convert_model.py"
    assert convert_file.exists(), f"Could not find convert_model.py at {convert_file}"
    os.system(
        f"python3 {convert_file} {output_path.with_suffix('.keras')} "
        f"{output_path.with_suffix('.json')}"
    )
