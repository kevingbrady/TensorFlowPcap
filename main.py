import time
import keras.utils
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Normalization, Dense, Input


columns = {
    'No': tf.int64,
    'src_ip': tf.float64,
    'dst_ip': tf.float64,
    'src_port': tf.int64,
    'dst_port': tf.int64,
    'protocol': tf.int64,
    'pkt_length': tf.int64,
    'info': tf.float64,
    'timestamp': tf.float64,
    'flow_duration': tf.float64,
    'flow_byts_s': tf.float64,
    'flow_pkts_s': tf.float64,
    'fwd_pkts_s': tf.float64,
    'bwd_pkts_s': tf.float64,
    'tot_fwd_pkts': tf.float64,
    'tot_bwd_pkts': tf.float64,
    'totlen_fwd_pkts': tf.float64,
    'totlen_bwd_pkts': tf.float64,
    'fwd_pkt_len_max': tf.float64,
    'fwd_pkt_len_min': tf.float64,
    'fwd_pkt_len_mean': tf.float64,
    'fwd_pkt_len_std': tf.float64,
    'bwd_pkt_len_max': tf.float64,
    'bwd_pkt_len_min': tf.float64,
    'bwd_pkt_len_mean': tf.float64,
    'bwd_pkt_len_std': tf.float64,
    'pkt_len_max': tf.float64,
    'pkt_len_min': tf.float64,
    'pkt_len_mean': tf.float64,
    'pkt_len_std': tf.float64,
    'pkt_len_var': tf.float64,
    'fwd_header_len': tf.int64,
    'bwd_header_len': tf.int64,
    'fwd_seg_size_min': tf.float64,
    'fwd_act_data_pkts': tf.float64,
    'flow_iat_mean': tf.float64,
    'flow_iat_max': tf.float64,
    'flow_iat_min': tf.float64,
    'flow_iat_std': tf.float64,
    'fwd_iat_tot': tf.float64,
    'fwd_iat_max': tf.float64,
    'fwd_iat_min': tf.float64,
    'fwd_iat_mean': tf.float64,
    'fwd_iat_std': tf.float64,
    'bwd_iat_tot': tf.float64,
    'bwd_iat_max': tf.float64,
    'bwd_iat_min': tf.float64,
    'bwd_iat_mean': tf.float64,
    'bwd_iat_std': tf.float64,
    'fwd_psh_flags': tf.int64,
    'bwd_psh_flags': tf.int64,
    'fwd_urg_flags': tf.int64,
    'bwd_urg_flags': tf.int64,
    'fin_flag_cnt': tf.int64,
    'syn_flag_cnt': tf.int64,
    'rst_flag_cnt': tf.int64,
    'psh_flag_cnt': tf.int64,
    'ack_flag_cnt': tf.int64,
    'urg_flag_cnt': tf.int64,
    'ece_flag_cnt': tf.int64,
    'down_up_ratio': tf.float64,
    'pkt_size_avg': tf.float64,
    'init_fwd_win_byts': tf.float64,
    'init_bwd_win_byts': tf.float64,
    'active_max': tf.float64,
    'active_min': tf.float64,
    'active_mean': tf.float64,
    'active_std': tf.float64,
    'idle_max': tf.float64,
    'idle_min': tf.float64,
    'idle_mean': tf.float64,
    'idle_std': tf.float64,
    'fwd_byts_b_avg': tf.float64,
    'fwd_pkts_b_avg': tf.float64,
    'bwd_byts_b_avg': tf.float64,
    'bwd_pkts_b_avg': tf.float64,
    'fwd_blk_rate_avg': tf.float64,
    'bwd_blk_rate_avg': tf.float64,
    'Target': tf.int64
}


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


if __name__ == '__main__':

    batch_size = 192
    column_names = list(columns.keys())
    csv_file = 'preprocessedData.csv'

    dataset_size = sum(1 for row in open(csv_file))
    print(dataset_size)

    split_size, remainder = divmod(dataset_size, batch_size)

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file,
        batch_size=split_size,
        column_names=column_names,
        column_defaults=columns.values(),
        label_name=column_names[-1],
        shuffle=True,
        ignore_errors=True,
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
        Normalization(axis=-1),
        Dense(1, activation='sigmoid')
    ])(features)

    model = keras.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)

    start = time.time()
    model.fit(
        train,
        epochs=10,
        batch_size=split_size,
        use_multiprocessing=True
    )

    #test_loss, test_accuracy = model.evaluate(test, batch_size=split_size, use_multiprocessing=True, verbose=1)
    #val_loss, val_accuracy = model.evaluate(validation, batch_size=split_size, use_multiprocessing=True, verbose=1)

    test_labels = np.concatenate([y for x, y in test], axis=0)
    test_results = np.array(model.predict(test, batch_size=split_size, use_multiprocessing=True))
    test_accuracy = 1 - np.mean(test_labels != test_results)

    val_labels = np.concatenate([y for x, y in validation], axis=0)
    val_results = np.array(model.predict(validation, batch_size=split_size, use_multiprocessing=True))
    val_accuracy = 1 - np.mean(val_labels != val_results)

    end = time.time()

    print("Test: %.3f \nValidation: %.3f" % (test_accuracy, val_accuracy))
    print("Run Time " + print_run_time(end - start))

    '''
    for batch, label in train.take(1):
        for key, value in batch.items():
            print(f"{key:20s}: {value}")
        print()
        print(f"{'label':20s}: {label}")
        
    '''







