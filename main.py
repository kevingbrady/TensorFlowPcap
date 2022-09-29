import time
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Input, Dropout
from src.DataManager import DataManager
from src.utils import print_run_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    csv_file = 'preprocessedData.csv'
    batch_size = 160  # TRAINING + TEST + VALIDATION BATCH SIZE

    manager = DataManager(csv_file, batch_size)
    print('Dataset Size: ', manager.dataset_size)
    print('Num Features: ', manager.num_features)

    # Check if new regularized data file needs to be created, otherwise use existing one
    normalized_csv_file = manager.get_normalized_data_file('l2')

    # Make Dataset from Normalized Data File and Shuffle Entries
    dataset = tf.data.experimental.make_csv_dataset(
        normalized_csv_file,
        batch_size=manager.split_size,
        column_names=manager.feature_names,
        column_defaults=manager.features['normalized'].values(),
        label_name=manager.feature_names[-1],
        sloppy=True,
        shuffle=True,
        num_parallel_reads=os.cpu_count(),
        num_epochs=1
    )

    # Split Data into Train, Test, Validation Sets
    train, validation, test = manager.get_dataset_partitions_tf(dataset)

    # Prefetch and Cache Data to help speed up training and evaluation
    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    # Build Model by setting up layers, starting with input layer to capture the features in the dataset
    inputs = {}
    feature_columns = []

    # Prepare Input Tensor ignoring The first and last columns of the dataframe ('No', 'Target')
    for name in manager.feature_names[1:-1]:
        #print(i)
        feature_columns.append(tf.feature_column.numeric_column(name))
        inputs[name] = Input(shape=(1,), name=name)

    features = tf.keras.layers.DenseFeatures(feature_columns=feature_columns)(inputs)

    # Build Model Layers
    x = Sequential([
        BatchNormalization(),
        Dense(manager.num_features, activation='relu'), #, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(manager.num_features * 2, activation='relu'), # kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(manager.num_features, activation='relu'), #, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output
    ])(features)

    # Compile Model with Adam Optimizer and Binary Crossentropy loss function
    model = keras.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', 'Precision', 'Recall'] #'AUC']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)

    # Train model using multiple epochs, with the validation
    # set being evaluated after each epoch to check for overfitting
    start = time.time()
    model.fit(
        train,
        epochs=50,
        validation_data=validation,
    )

    # Evaluate model using test data that has been held out
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test, verbose=0)
    end = time.time()

    print("Test Loss: %.3f\n Test Accuracy: %.3f\n Test Precision: %.3f\n Test Recall: %.3f" % (test_loss, test_accuracy, test_precision, test_recall))
    print("Run Time " + print_run_time(end - start))
