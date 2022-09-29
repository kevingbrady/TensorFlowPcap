import time
import os
from .utils import print_run_time
from .CsvReader import CsvReader
from .metadata.data_columns import columns_normalized

class CsvNormalizer(CsvReader):

    normalized_filename = ''
    columns = list(columns_normalized.keys())

    def __init__(self, file):
        super(CsvNormalizer, self).__init__(file)

    def create_normalized_csvfile(self, method):

        assert os.path.exists(self.filename)
        assert method == 'l2' or method == 'zscore'
        self.normalized_filename = os.path.splitext(self.filename)[0] + "_normalized_" + method + ".csv"

        if self.gpu_available > 0:
            data_generator = self.read_csv_gpu()
            if method == 'l2':
                self._normalize_dataset_l2_gpu(data_generator)
            else:
                self._normalize_dataset_zscore_gpu(data_generator)
        else:
            data_generator = self.read_csv_cpu()
            if method == 'l2':
                self._normalize_dataset_l2(data_generator)
            else:
                self._normalize_dataset_zscore(data_generator)

    def write_chunk(self, chunk, header, mode):

        with open(self.normalized_filename, mode) as outfile:
            chunk.to_csv(outfile, header=header, index=False, columns=self.columns)

    # Normalize Entire Dataset in Batches using L2 Normalization
    def _normalize_dataset_l2(self, data_generator):

        i = 0
        s = time.time()

        for chunk in data_generator:
            header = (True if i == 0 else False)
            mode = ('w' if i == 0 else 'a')

            chunk = chunk.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.name not in ('No', 'Target') else x,
                                axis=0)
           # chunk.to_csv(self.normalized_filename, header=header, mode=mode, index=False, na_rep='0.0',
            #             columns=data_columns.columns.keys())
            self.write_chunk(chunk, header, mode)
            i += 1
        e = time.time()
        print("CPU Data Normalization Completed In: " + print_run_time(e - s))

    # Normalize Entire Dataset in Batches Using ZScore Normalization
    def _normalize_dataset_zscore(self, data_generator):

        i = 0
        s = time.time()

        for chunk in data_generator:
            header = (True if i == 0 else False)
            mode = ('w' if i == 0 else 'a')

            chunk = chunk.apply(lambda x: (x - x.mean()) / x.std() if x.name not in ('No', 'Target') else x, axis=0)
            #chunk.to_csv(self.normalized_filename, header=header, mode=mode, index=False, na_rep='0.0',
            #             columns=data_columns.columns.keys())
            self.write_chunk(chunk, header, mode)
            i += 1

        e = time.time()
        print("CPU Data Normalization Completed In: " + print_run_time(e - s))

    # Normalize Entire Dataset in Batches using L2 Normalization over GPU
    def _normalize_dataset_l2_gpu(self, data_generator, ddof=1.0):

        i = 0
        s = time.time()

        for chunk in data_generator:
            header = (True if i == 0 else False)
            mode = ('w' if i == 0 else 'a')

            for col in chunk:
                if col not in ('No', 'Target'):
                    # chunk[col] = (chunk[col] - chunk[col].min()) / (ddof - (chunk[col].max() - chunk[col].min()))
                    chunk[col] = (chunk[col] - chunk[col].min()) / ((chunk[col].max() - chunk[col].min())) if (
                                chunk[col].max() - chunk[col].min()) else 0.0
                    # chunk[col] = 0 if (chunk[col].max() - chunk[col].min()) == 0 else (chunk[col] - chunk[col].min()) / ((chunk[col].max() - chunk[col].min()))

            #with open(self.normalized_filename, mode) as outfile:
            #    chunk.to_csv(outfile, header=header, index=False, columns=data_columns.columns.keys())
            self.write_chunk(chunk, header, mode)
            i += 1

        e = time.time()
        print("GPU Data Normalization Completed In:  " + print_run_time(e - s))

    # Normalize Entire Dataset in Batches Using ZScore Normalization over GPU
    def _normalize_dataset_zscore_gpu(self, data_generator, ddof=1.0):

        i = 0
        s = time.time()

        for chunk in data_generator:
            header = (True if i == 0 else False)
            mode = ('w' if i == 0 else 'a')

            for col in chunk:
                if col not in ('No', 'Target'):
                    # chunk[col] = (chunk[col] - chunk[col].mean()) / (ddof - chunk[col].std())
                    chunk[col] = (chunk[col] - chunk[col].mean()) / (chunk[col].std(ddof=0)) if chunk[col].std(
                        ddof=0) else 0.0
                    # chunk[col] = 0 if chunk[col].std() == 0 else (chunk[col] - chunk[col].mean()) / (chunk[col].std())

            #with open(self.normalized_filename, mode) as outfile:
            #    chunk.to_csv(outfile, header=header, index=False)
            self.write_chunk(chunk, header, mode)
            i += 1

        e = time.time()
        print("GPU Data Normalization Completed In:  " + print_run_time(e - s))