import time
import keras.utils
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization, Normalization, Dense, Input, Dropout
import json
import hashlib
import data_columns

# Import Column Types from Data Columns file
columns = data_columns.columns_normalized


def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))


# Split Dataset into Train, Test, and Validation Sets. Default Split is 80/10/10
def get_dataset_partitions_tf(ds, ds_size, split_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    train_size = int((train_split * ds_size) / split_size)
    val_size = int((val_split * ds_size) / split_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


# Normalize Entire Dataset in Batches using L2 Normalization
def _normalize_dataset_l2(csv_file):

    csv_data = pd.read_csv(csv_file, chunksize=112000, iterator=True)
    i = 0

    for chunk in csv_data:
        header = (True if i == 0 else False)
        mode = ('w' if i == 0 else 'a')
        chunk = chunk.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.name not in ('No', 'Target') else x, axis=0)
        chunk.to_csv('preprocessedData_normalized_l2.csv', header=header, mode=mode, index=False, float_format="%.5f", na_rep='0.0', columns = columns.keys())
        i += 1


# Normalize Entire Dataset in Batches Using ZScore Normalization
def _normalize_dataset_zscore(csv_file):

    csv_data = pd.read_csv(csv_file, chunksize=112000, iterator=True)
    i = 0

    for chunk in csv_data:
        header = (True if i == 0 else False)
        mode = ('w' if i == 0 else 'a')
        chunk = chunk.apply(lambda x: (x - x.mean()) / x.std() if x.name not in ('No', 'Target') else x, axis=0)
        chunk.to_csv('preprocessedData_normalized_zscore.csv', header=header, mode=mode, index=False, float_format="%.5f", na_rep='0.0', columns = columns.keys())
        i += 1


# Normalize Dataset using defined method (l2 or zscore). If the Normalized Data file already exists and has not been
# updated return the file without performing any unnecessary work
def check_data(csv_file, method):

    assert method == 'l2' or method == 'zscore'
    file = "preprocessedData_normalized_" + method + ".csv"

    if os.path.exists(file):

        with open("data_file_hashes.json") as data_file:

            data_file_hashes = json.load(data_file)
            file_hash = hashlib.md5(open(file, 'rb').read()).hexdigest()

            if file_hash == data_file_hashes[method]:
                print(file + " up to date, continuing ...")
                return file

        data_file_hashes[method] = file_hash
        json.dump(data_file_hashes, data_file)

    print("No normalized data file found, creating new file ...")
    if method == 'l2':
        _normalize_dataset_l2(csv_file)
    if method == 'zscore':
        _normalize_dataset_zscore(csv_file)

    return file


if __name__ == '__main__':

    csv_file = 'preprocessedData.csv'
    normalized_csv_file = check_data(csv_file, 'zscore')

    batch_size = 80            # TRAINING + TEST + VALIDATION (80 , 10, 10)
                               # Training Batch Size / (Percentage of Data Allocated for Training)

    column_names = list(columns.keys())
    dataset_size = sum(1 for row in open(normalized_csv_file))
    print('Dataset Size: ', dataset_size)
    print('Num Features: ', len(columns))

    # Use Batch Size to Determine the Size of Each Batch of Data, with the last batch consisting of the remainder of data
    split_size, remainder = divmod(dataset_size, batch_size)

    # Make Dataset from Normalized Data File and Shuffle Entries
    dataset = tf.data.experimental.make_csv_dataset(
        normalized_csv_file,
        batch_size=split_size,
        column_names=column_names,
        column_defaults=columns.values(),
        label_name=column_names[-1],
        sloppy=True,
        shuffle=True,
        shuffle_buffer_size=10000,
        num_parallel_reads=os.cpu_count(),
        num_epochs=1
    )

    # Split Data into Train, Test, Validation Sets
    train, validation, test = get_dataset_partitions_tf(dataset, dataset_size, split_size)

    # Prefetch and Cache Data to help speed up training and evaluation
    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    # Build Model by setting up layers, starting with input layer to capture the features in the dataset
    inputs = {}
    feature_columns = []

    for i in column_names[:-1]:
        feature_columns.append(tf.feature_column.numeric_column(i))
        inputs[i] = Input(shape=(1, ), name=i)

    features = tf.keras.layers.DenseFeatures(feature_columns=feature_columns)(inputs)

    x = Sequential([
        BatchNormalization(),
        Dense(len(columns), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(columns) * 2, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(columns), activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')   # Output
    ])(features)

    # Compile Model with Adam Optimizer and Binary Crossentropy loss function
    model = keras.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)

    # Train model using multiple epochs, with the validation set being evaluated after each epoch to check for overfitting
    start = time.time()
    model.fit(
        train,
        epochs=100,
        batch_size=(batch_size / 0.8) * 2,
        validation_data=validation,
    )

    # Evaluate model using test data that had been held out
    test_loss, test_accuracy = model.evaluate(test, batch_size=(batch_size / 0.8) * 2, verbose=0)
    end = time.time()

    print("Test Loss: %.3f\n Test Accuracy: %.3f" % (test_loss, test_accuracy))
    print("Run Time " + print_run_time(end - start))




