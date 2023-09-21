from pathlib import Path
from cal_ratio_trainer.config import ConvertConfig
from cal_ratio_trainer.reporting.evaluation_utils import load_trained_model
from cal_ratio_trainer.utils import find_training_result


def convert_file(c: ConvertConfig):
    # Load up the model and weights for the run.
    assert c.run_to_convert is not None
    model_path = find_training_result(c.run_to_convert.name, c.run_to_convert.run)
    model = load_trained_model(model_path, c.run_to_convert.epoch)

    # Now write it out!
    assert c.output_json is not None
    output_path = Path(c.output_json)

    # Make sure the directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write out as keras output
    model.model.save(output_path.with_suffix(".keras"))

    # Next, convert it to fdeep format.
