import tensorflow as tf
import pandas as pd
import sys
import importlib.util
from .metadata.data_columns import columns

GPUs = tf.config.list_physical_devices('GPU')
cudf = importlib.util.find_spec('cudf')

if cudf is not None:
    import cudf


#if sys.platform.__contains__('linux') and len(GPUs) > 0:
#    import cudf


class CsvReader:

    CHUNKSIZE = 1000000

    def __init__(self, filename):

        self.filename = filename
        self.gpus_available = len(GPUs)
        self.use_cudf = (cudf is not None)

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