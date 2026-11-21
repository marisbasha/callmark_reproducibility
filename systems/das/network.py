"""Use of the DAS package (written for Keras 2) with Keras 3, and the model location."""
import keras

from benchmark import config

MODEL_NAME = "das"


def model_dir(size):
    """Return models/das/<size> of the data root."""
    return config.DATA / "models" / "das" / str(size)


def model_prefix(size):
    """Return the path to which DAS appends _model.h5 and _params.yaml."""
    return str(model_dir(size) / MODEL_NAME)


def compile_without_sample_weight_mode(original_compile):
    """Return a Model.compile that drops the argument sample_weight_mode."""
    def compile(self, *arguments, sample_weight_mode=None, **keywords):
        return original_compile(self, *arguments, **keywords)
    return compile


def prepare_keras():
    """Make Keras 3 accept the two Keras 2 conventions that DAS relies on.

    DAS compiles its model with sample_weight_mode="temporal". Keras 3 has no such argument
    and takes the weighting from the shape of the sample weights, so the argument is dropped.
    DAS normalizes channels in Lambda layers; Keras 3 loads a saved model with such layers
    only if the function is registered."""
    keras.Model.compile = compile_without_sample_weight_mode(keras.Model.compile)
    from das.tcn.tcn_new import channel_normalization
    keras.saving.get_custom_objects()["channel_normalization"] = channel_normalization
