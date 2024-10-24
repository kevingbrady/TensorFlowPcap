import tensorflow as tf
import keras


class InputLayer:

    inputs = {}
    num_features = 0

    def get_input_tensor(self):

        return keras.layers.Concatenate()(list(self.inputs.values()))

    def __init__(self, manager):

        self.features = [x for x in manager.feature_names if x != 'Target']
        self.num_features = len(self.features)

        for key in self.features:
            self.inputs[key] = keras.layers.Input(shape=(1,), name=key)
