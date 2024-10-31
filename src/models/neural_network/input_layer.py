import keras


class InputLayer:

    inputs = []
    num_features = 0

    def get_input_tensor(self):

        return keras.layers.Concatenate()(self.inputs)

    def __init__(self, manager):

        self.features = {x: y for x, y in manager.features.items() if x != 'Target'}
        self.num_features = len(self.features)

        for key, value in self.features.items():
            self.inputs.append(keras.layers.Input(shape=(1,), name=key, dtype=value))
