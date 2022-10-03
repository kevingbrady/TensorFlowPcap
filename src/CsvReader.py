import tensorflow as tf
import pandas as pd
import sys
from .metadata.data_columns import columns

GPUs = tf.config.list_physical_devices('GPU')

if sys.platform.__contains__('linux') and len(GPUs) > 0:
    import cudf


class CsvReader:

    CHUNKSIZE = 1000000

    def __init__(self, filename):

        self.filename = filename
        self.gpus_available = len(GPUs)
        self.use_cudf = (sys.platform.__contains__('linux') and self.gpus_available > 0)

    def read_file(self):

        if self.gpus_available > 0 and self.use_cudf:
            return self.read_csv_gpu()
        else:
            return self.read_csv_cpu()

    def read_csv_cpu(self):

        return pd.read_csv(self.filename,
                           chunksize=self.CHUNKSIZE,
                           iterator=True)

    def read_csv_gpu(self):

        assert self.gpus_available > 0

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

'''
if __name__ == '__main__':

    csv_file = CsvReader('../preprocessedData.csv')
    data_generator = csv_file.read_file()

    for chunk in data_generator:
        print(chunk['No'].values[-5:])
        print('\n\n')
'''
