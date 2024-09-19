import tensorflow as tf
import keras


class InputLayer:

    inputs = {}
    feature_columns = []
    num_features = 0
    jit_compile = False
    exclude_features = [
        'Target',
        'No',
        'timestamp',
        #'src_ip',
        #'dst_ip',
        'fwd_seg_size_avg',
        'bwd_seg_size_avg',
        'cwe_flag_count',
        'subflow_fwd_pkts',
        'subflow_bwd_pkts',
        'subflow_fwd_byts',
        'subflow_bwd_byts'
    ]

    def get_input_tensor(self):

        return keras.layers.Concatenate(axis=1)(list(self.inputs.values()))

    def __init__(self, manager):

        features = manager.features['data']

        self.jit_compile = manager.jit_compile
        self.features = {x: y for x, y in features.items() if x not in self.exclude_features}
        self.num_features = len(self.features)

        for key, value in self.features.items():
            self.inputs[key] = keras.layers.Input(shape=(1,), name=key)
