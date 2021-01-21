import pickle
import numpy as np


def pickle_save(obj, path):
    with open(path + '.pyc', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path + '.pyc', 'rb') as f:
        obj = pickle.load(f, encoding='bytes')
    return obj


def numpy_save(obj, path):
    with open(path + '.npy', 'wb') as f:
        np.save(f, obj)


def numpy_load(path):
    with open(path + '.npy', 'rb') as f:
        obj = np.load(f)
    return obj
