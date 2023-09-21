from cal_ratio_trainer.common.trained_model import load_model


def dump_model(model_name: str):
    """
    Loads a saved Keras model from disk and prints a summary of its architecture.

    Args:
        model_name (str): The name of the saved model file to load.

    Returns:
        None
    """
    model = load_model(model_name)
    print(model.model.summary())
