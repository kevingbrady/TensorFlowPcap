import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns
from matplotlib.colors import ListedColormap

CHUNK_SIZE = 100000
column_names = ['No', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
       'pkt_length', 'info', 'timestamp', 'flow_duration', 'flow_byts_s',
       'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts',
       'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max',
       'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
       'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean',
       'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min', 'pkt_len_mean',
       'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len',
       'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean',
       'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot',
       'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std',
       'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean',
       'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
       'bwd_urg_flags', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt',
       'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt',
       'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts',
       'init_bwd_win_byts', 'active_max', 'active_min', 'active_mean',
       'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std',
       'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg',
       'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'Target']


def get_dataset_partitions(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':

        #dataSet = tf.data.experimental.make_csv_dataset('/home/kgb/PcapPreprocessor/preprocessedData.csv', column_names=column_names, label_name='Target', num_epochs=1, batch_size=12, na_value=None)
        dataset = pd.read_csv('/home/kgb/PcapPreprocessor/preprocessedData.csv', names=column_names)

        labels = dataset['Target']
        dataset.drop('Target')
        train_set, validation_set, test_set = get_dataset_partitions(dataset)
        print(len(train_set))
