import os
import pickle as pkl

from metric_bandits.paths import OBJECT_DUMP_PATH


def save_object(obj, filename):
    """
    Saves the object to filename from object_dump folder.
    """
    filename = os.path.join(OBJECT_DUMP_PATH, filename + ".pkl")
    with open(filename, "wb") as output:
        pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    Loads the object from filename from object_dump folder.
    """
    filename = os.path.join(OBJECT_DUMP_PATH, filename + ".pkl")
    with open(filename, "rb") as input:
        obj = pkl.load(input)
    return obj
