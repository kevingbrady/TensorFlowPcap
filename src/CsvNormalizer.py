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
            self._normalize_dataset(data_generator, method, 'GPU')
        else:
            data_generator = self.read_csv_cpu()
            self._normalize_dataset(data_generator, method, 'CPU')

    def _write_chunk(self, chunk, header, mode):

        with open(self.normalized_filename, mode) as outfile:
            chunk.to_csv(outfile, header=header, index=False, columns=self.columns)

    # Normalize Entire Dataset in Batches using L2 Normalization (CPU or GPU)
    def _normalize_dataset(self, data_generator, method, proc):

        i = 0
        s = time.time()

        for chunk in data_generator:
            header = (True if i == 0 else False)
            mode = ('w' if i == 0 else 'a')

            for col in chunk:
                if col not in ('No', 'Target'):

                    chunk[col] = self.normalize(chunk[col], method)

            self._write_chunk(chunk, header, mode)
            i += 1

        e = time.time()
        print(proc + " Data Normalization Completed In: " + print_run_time(e - s))

    # Normalize using l2 or zscore
    @staticmethod
    def normalize(x, method):
        if method == 'l2':
            return (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) else 0.0
        elif method == 'zscore':
            return (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0.0