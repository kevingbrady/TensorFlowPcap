import tensorflow as tf


class InputLayer:

    inputs = {}
    feature_columns = []
    num_features = 0
    jit_compile = False

    def __call__(self, manager):

        features = manager.feature_names
        self.jit_compile = manager.jit_compile

        for name in features:
            if name not in ('No', 'Target'):
                self.num_features += 1
                self.feature_columns.append(tf.feature_column.numeric_column(name))
                self.inputs[name] = tf.keras.layers.Input(shape=1, name=name)

        return tf.keras.layers.DenseFeatures(feature_columns=self.feature_columns, name='Input')(self.inputs)