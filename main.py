import time
import keras.utils
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization, Normalization, Dense, Input, Dropout
import hashlib
import data_columns

columns = data_columns.columns_normalized

def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))


def get_dataset_partitions_tf(ds, ds_size, split_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    train_size = int((train_split * ds_size) / split_size)
    val_size = int((val_split * ds_size) / split_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def normalize_dataset_l2(csv_file):

    csv_data = pd.read_csv(csv_file, chunksize=112000, iterator=True)
    i = 0

    for chunk in csv_data:
        header = (True if i == 0 else False)
        mode = ('w' if i == 0 else 'a')
        chunk = chunk.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.name not in ('No', 'Target') else x, axis=0)
        chunk.to_csv('preprocessedData_normalized_l2.csv', header=header, mode=mode, index=False, float_format="%.5f", na_rep='0.0', columns = columns.keys())
        i += 1


def normalize_dataset_zscore(csv_file):

    csv_data = pd.read_csv(csv_file, chunksize=112000, iterator=True)
    i = 0

    for chunk in csv_data:
        header = (True if i == 0 else False)
        mode = ('w' if i == 0 else 'a')
        chunk = chunk.apply(lambda x: (x - x.mean()) / x.std() if x.name not in ('No', 'Target') else x, axis=0)
        chunk.to_csv('preprocessedData_normalized_zscore.csv', header=header, mode=mode, index=False, float_format="%.5f", na_rep='0.0', columns = columns.keys())
        i += 1


def check_data(csv_file, method):

    assert method == 'l2' or method == 'zscore'
    file = "preprocessedData_normalized_" + method + ".csv"

    if os.path.exists(file):

        l2_hash = 'bd4240ad21ba6510a9ccabf084389a12'
        zscore_hash = '24688d0237eef73c302c39cbed7010cb'
        file_hash = hashlib.md5(open(file,'rb').read()).hexdigest()

        if file_hash == (l2_hash if method == 'l2' else zscore_hash):
            print(file + " up to date, continuing ...")
            return file

    print("No normalized data file found, creating new file ...")
    if method == 'l2':
        normalize_dataset_l2(csv_file)
    if method == 'zscore':
        normalize_dataset_zscore(csv_file)

    return file


if __name__ == '__main__':

    csv_file = 'preprocessedData.csv'
    normalized_csv_file = check_data(csv_file, 'zscore')

    batch_size = 80            # TRAINING + TEST + VALIDATION (80 , 10, 10)
    column_names = list(columns.keys())

    dataset_size = sum(1 for row in open(normalized_csv_file))
    print('Dataset Size: ', dataset_size)
    print('Num Features: ', len(columns))

    split_size, remainder = divmod(dataset_size, batch_size)

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

    train, validation, test = get_dataset_partitions_tf(dataset, dataset_size, split_size)

    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)


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

    model = keras.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)

    start = time.time()
    model.fit(
        train,
        epochs=50,
        batch_size=(batch_size / 0.8) * 2,
        validation_data=validation,
    )

    #test_loss, test_accuracy = model.evaluate(test, batch_size=split_size, use_multiprocessing=True, verbose=1)
    #val_loss, val_accuracy = model.evaluate(validation, batch_size=split_size, use_multiprocessing=True, verbose=1)


    test_loss, test_accuracy = model.evaluate(test, batch_size=(batch_size / 0.8) * 2, verbose=0)
    end = time.time()

    print("Test Loss: %.3f\n Test Accuracy: %.3f" % (test_loss, test_accuracy))
    print("Run Time " + print_run_time(end - start))




