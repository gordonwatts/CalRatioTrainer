import os
from pathlib import Path
from cal_ratio_trainer.common.trained_model import load_trained_model_from_training
from cal_ratio_trainer.config import ConvertTrainingConfig
from cal_ratio_trainer.utils import find_training_result


def convert_file(c: ConvertTrainingConfig):
    # Load up the model and weights for the run.
    assert c.run_to_convert is not None
    model_path = find_training_result(c.run_to_convert.name, c.run_to_convert.run)
    model = load_trained_model_from_training(model_path, c.run_to_convert.epoch)

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
