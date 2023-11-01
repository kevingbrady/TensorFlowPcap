import tensorflow as tf
import sys
from src.CsvReader import CsvReader
from docker_info import DOCKER_PREFIX

GPUs = tf.config.list_physical_devices('GPU')

if sys.platform.__contains__('linux') and len(GPUs) > 0:
    import cudf


if __name__ == '__main__':

    print(GPUs)
    csv_file = CsvReader(DOCKER_PREFIX + 'preprocessedData.csv')
    data_generator = csv_file.read_file()

    for chunk in data_generator:
        print(chunk['No'].values[-5:])
        print('\n\n')
