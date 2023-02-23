import tensorflow as tf
from src.CsvReader import CsvReader
import sys

GPUs = tf.config.list_physical_devices('GPU')

if sys.platform.__contains__('linux') and len(GPUs) > 0:
    import cudf


if __name__ == '__main__':

    print(GPUs)
    csv_file = CsvReader('preprocessedData.csv')
    data_generator = csv_file.read_file()

    for chunk in data_generator:
        print(chunk['No'].values[-5:])
        print('\n\n')
