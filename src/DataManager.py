import os
import json
import hashlib
import pickle
import tensorflow as tf
from src.models.neural_network.input_layer import InputLayer
from src.metadata.data_columns import columns, columns_normalized
from src.CsvNormalizer import CsvNormalizer
from src.ClassAnalyzer import ClassAnalyzer
from docker_info import DOCKER_PREFIX


class DataManager:

    file = ''
    file_hash = ''
    dataset_size = 0
    jit_compile = False
    split_size = 0
    remainder = 0
    batch_size = 0
    normalization_method = 'l2'

    def __init__(self,
                 file,
                 batch_size,
                 training_split_percentage=0.75,
                 validation_split_percentage=0.125,
                 test_split_percentage=0.125
                 ):

        self.file = file
        self.exclude_features = ['src_ip', 'dst_ip']
        self.features = {x: y for x, y in columns.items() if x not in self.exclude_features}
        self.num_features = len(self.features)

        analyzer = self.get_class_analyzer()

        # Get Dataset Size by Opening Csv File and Counting rows
        self.dataset_size = analyzer.dataset_size
        self.batch_size = (batch_size * 1.25)    # Multiply by 1.25 to account for validation and test batches

        # Default Split is 75 / 12.5 / 12.5
        self.training_split_percentage = training_split_percentage
        self.validation_split_percentage = validation_split_percentage
        self.test_split_percentage = test_split_percentage

        assert (self.training_split_percentage + self.validation_split_percentage + self.test_split_percentage) == 1

        # Use Batch Size to Determine the Size of Each Batch of Data,
        # with the last batch consisting of the remainder of data
        self.split_size, self.remainder = divmod(self.dataset_size, self.batch_size)

        self.training_steps = 0
        self.validation_steps = 0

        analyzer.print_class_analysis_output()
        print('Num Features: ', self.num_features - 1)         # Subtract 1 feature for Target column

    def get_class_analyzer(self):

        if self.check_file_hash(self.file) and os.path.exists(DOCKER_PREFIX + "src/metadata/class_analyzer.pkl"):
            with open(DOCKER_PREFIX + "src/metadata/class_analyzer.pkl", 'rb') as h:
                analyzer = pickle.load(h)

        else:
            analyzer = ClassAnalyzer(self.file)
            self.write_metadata_file(self.file)
            with open(DOCKER_PREFIX + "src/metadata/class_analyzer.pkl", 'wb') as h:
                pickle.dump(analyzer, h)

        return analyzer

    # Normalize Dataset using defined method (l2 or zscore). If the Normalized Data file already exists and has not been
    # updated return the file without performing any unnecessary work
    def get_normalized_data_file(self, method):

        normalized_file = os.path.splitext(self.file)[0] + "_normalized_" + method + ".csv"
        csv_normalizer = CsvNormalizer(self.file)

        if csv_normalizer.gpus_available > 0:
            self.jit_compile = False

        if self.check_file_hash(normalized_file):
            print(normalized_file + " up to date, continuing ...")
            return normalized_file
        else:
            print("No normalized data file found, creating new file ...")
            csv_normalizer.create_normalized_csvfile(ignore_features=['src_ip', 'dst_ip', 'Target'], method=method)
            self.write_metadata_file(normalized_file)
            return normalized_file

    # Split Dataset into Train, Test, and Validation Sets.
    def get_dataset_partitions_tf(self, ds):

        train_size = int((self.training_split_percentage * self.dataset_size) / self.split_size)
        val_size = int((self.validation_split_percentage * self.dataset_size) / self.split_size)

        self.training_steps = int(train_size / self.split_size)
        self.validation_steps = int(val_size / self.split_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        print("Batch: ", int(self.split_size))
        print("Training : ", int(train_size * self.split_size))
        print("Validation: ", int(val_size * self.split_size))
        print("Testing: ", int(self.test_split_percentage * self.dataset_size))

        return train_ds, val_ds, test_ds

    # Create Tensorflow Dataset based on arguments that are conditional to which model is being used for training
    def load_dataset(self, csvfile, model_name):

        args = {
            'file_pattern': csvfile,
            'batch_size': int(self.split_size),
            'select_columns': self.features.keys(),
            'column_defaults': self.features.values(),
            'label_name': 'Target',
            'header': True,
            #'shuffle': True,
            #'sloppy': True,
            #'num_parallel_reads': os.cpu_count(),
            'num_epochs': 1,
        }
        '''
        if model_name in ('DeepNeuralNet', 'LogisticRegression'):
            normalized_csvfile = self.get_normalized_data_file(self.normalization_method)
            args.update({
                'file_pattern': normalized_csvfile,
                'column_defaults': [y for x, y in columns_normalized.items() if x not in self.exclude_features],
            })
        '''
        return tf.data.experimental.make_csv_dataset(**args)

    @staticmethod
    def write_metadata_file(file):

        with open(DOCKER_PREFIX + "src/metadata/data_file_hashes.json", "r+") as data_file:
            file_hash = hashlib.md5(open(file, 'rb').read()).hexdigest()
            data_file_hashes = json.load(data_file)
            data_file.seek(0)
            data_file_hashes[file] = file_hash
            json.dump(data_file_hashes, data_file)


    @staticmethod
    def check_file_hash(file):

        with open(DOCKER_PREFIX + "src/metadata/data_file_hashes.json", "r+") as data_file:

            data_file_hashes = json.load(data_file)

            if file in data_file_hashes and os.path.exists(file):

                file_hash = hashlib.md5(open(file, 'rb').read()).hexdigest()

                if file_hash == data_file_hashes[file]:
                    return True

            return False
