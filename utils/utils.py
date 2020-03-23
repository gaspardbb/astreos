import os
import pickle
from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def save_obj(path, obj, name):
    """ Save object obj in file path/name.pkl """
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    """ Load object from file path/name.pkl """
    with open(os.path.join(path, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
