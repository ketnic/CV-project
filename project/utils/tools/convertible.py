import os


def list_of_tuples_to_dict(list):
    d = {}
    for x, y in list:
        d.setdefault(x, []).append(y)
    return d


def join_to_path(iter):
    return os.path.sep.join(iter)


def flatten(matrix):
    return [item for sublist in matrix for item in sublist]