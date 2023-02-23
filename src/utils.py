import numpy as np


def get_class_probabilities(data, model):
    return np.array(list(zip(1-model.predict(data), model.predict(data))))


def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))
