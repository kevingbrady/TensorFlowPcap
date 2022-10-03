import os
import json
import hashlib
from .metadata.data_columns import columns, columns_normalized
from .CsvNormalizer import CsvNormalizer


class DataManager:

    file = ''
    dataset_size = 0
    num_features = len(columns)
    feature_names = list(columns.keys())
    features = {'data': columns, 'normalized': columns_normalized}
    split_size = 0
    remainder = 0
    batch_size = 0

    def __init__(self, file, batch_size):

        # Get Dataset Size by Opening Csv File and Counting rows
        self.dataset_size = sum(1 for row in open(file))
        self.batch_size = batch_size
        self.file = file

        # Use Batch Size to Determine the Size of Each Batch of Data,
        # with the last batch consisting of the remainder of data
        self.split_size, self.remainder = divmod(self.dataset_size, self.batch_size)

    # Normalize Dataset using defined method (l2 or zscore). If the Normalized Data file already exists and has not been
    # updated return the file without performing any unnecessary work
    def get_normalized_data_file(self, method):

        file = os.path.splitext(self.file)[0] + "_normalized_" + method + ".csv"

        with open("src/metadata/data_file_hashes.json", "r+") as data_file:

            data_file_hashes = json.load(data_file)

            if os.path.exists(file):

                file_hash = hashlib.md5(open(file, 'rb').read()).hexdigest()

                if file_hash == data_file_hashes[method]:
                    print(file + " up to date, continuing ...")
                    return file

            print("No normalized data file found, creating new file ...")

            csv_normalizer = CsvNormalizer(self.file)
            csv_normalizer.create_normalized_csvfile(ignore_features=('No', 'Target'), method=method)

            data_file.seek(0)
            data_file_hashes[method] = hashlib.md5(open(file, 'rb').read()).hexdigest()
            json.dump(data_file_hashes, data_file)

        return file

    # Split Dataset into Train, Test, and Validation Sets. Default Split is 75/12.5/12.5
    def get_dataset_partitions_tf(self, ds, train_split=0.8, val_split=0.1, test_split=0.1):
        assert (train_split + test_split + val_split) == 1

        train_size = int((train_split * self.dataset_size) / self.split_size)
        val_size = int((val_split * self.dataset_size) / self.split_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds



