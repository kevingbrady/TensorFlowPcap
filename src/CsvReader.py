import tensorflow as tf
import cudf
import pandas as pd
from .metadata.data_columns import columns


class CsvReader:

    CHUNKSIZE = 1000000
    BYTE_RANGE = 1e9

    def __init__(self, filename):

        self.filename = filename
        self.gpu_available = len(tf.config.list_physical_devices('GPU'))

    def read_file(self):

        if self.gpu_available:
            return self.read_csv_gpu()
        else:
            return self.read_csv_cpu()

    def read_csv_cpu(self):

        return pd.read_csv(self.filename,
                           chunksize=self.CHUNKSIZE,
                           dtype=columns.values(),
                           iterator=True)

    def read_csv_gpu(self):

        assert self.gpu_available > 0

        offset = 0
        while True:

            try:
                chunk = cudf.read_csv(self.filename,
                                      names=list(columns.keys()),
                                      dtype=columns.values(),
                                      skiprows=(self.CHUNKSIZE * offset),
                                      nrows=self.CHUNKSIZE,
                                      header=(offset == 0),
                                      use_python_file_object=True)
                offset += 1
                #print(offset * self.CHUNKSIZE)
                yield chunk

            except RuntimeError:
                break


if __name__ == '__main__':

    csv_file = CsvReader('../preprocessedData.csv')
    data_generator = csv_file.read_file()

    for chunk in data_generator:
        print(chunk['No'].values[-5:])
        print('\n\n')
