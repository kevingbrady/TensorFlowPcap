

def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))


def print_results(results_dict):

    print("\n")
    print({print(key + ': ' + str(value) for key, value in results_dict.items())})
