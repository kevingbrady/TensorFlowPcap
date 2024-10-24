import os
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import BatchNormalization, Dense, Dropout, LayerNormalization, UnitNormalization, GaussianDropout
from src.models.neural_network.input_layer import InputLayer
from docker_info import DOCKER_PREFIX


class NeuralNet:

    name = "DeepNeuralNet"
    model_filepath = DOCKER_PREFIX + 'src/models/neural_network/deep_neural_network/DeepNeuralNet'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    #loss = tf.keras.losses.SquaredHinge()
    #loss = tf.keras.losses.Hinge()
    metrics = ['accuracy', 'BinaryAccuracy', 'Precision', 'Recall']
    jit_compile = False
    epochs = 30

    def __init__(self, manager):

        if not os.path.exists(self.model_filepath):
            os.mkdir(self.model_filepath)

        self.input_layer = InputLayer(manager)
        self.input_tensor = self.input_layer.get_input_tensor()

        self.classifier = Sequential([
            BatchNormalization(),
            #LayerNormalization(),
            #UnitNormalization(),
            Dense(self.input_tensor.shape[1], activation='leaky_relu', kernel_initializer='glorot_uniform'),
            #BatchNormalization(),
            #Dropout(0.2),
            Dense(self.input_tensor.shape[1] * 2, activation='leaky_relu', kernel_initializer='glorot_uniform'),
            #BatchNormalization(),
            #Dropout(0.4),
            Dense(self.input_tensor.shape[1], activation='leaky_relu', kernel_initializer='glorot_uniform'),
            #BatchNormalization(),
            #Dropout(0.2),
            Dense(1, activation='sigmoid')  # Output
        ], name='Network')(self.input_tensor)

    def save_model(self, model):
        model.save(self.model_filepath + '.keras')

    def save_model_diagram(self, model):
        tf.keras.utils.plot_model(
            model.get_layer('Network'),
            to_file=DOCKER_PREFIX + 'src/models/neural_network/deep_neural_network/' + self.name + '.png',  # saving
            show_layer_activations=True,
            show_shapes=True,
            show_layer_names=True,  # show shapes and layer name
            expand_nested=True  # will show nested block
        )

    def __call__(self):

        model = tf.keras.Model(
            name=self.name,
            inputs=self.input_layer.inputs,
            outputs=self.classifier
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            jit_compile=self.jit_compile
        )

        #print(model.summary())
        return model
