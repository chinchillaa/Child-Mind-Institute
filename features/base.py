import argparse
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    """
    Simple timer context manager.

    Parameters
    ----------
    name : str
        Name to be printed when the timer starts and ends.

    Examples
    --------
    >>> with timer('my_func'):
    ...     my_func()
    [my_func] start
    [my_func] done in 0.1 s
    """
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def get_arguments():
    """
    Parse command line arguments for running feature extraction.

    This function parses the following options:

    * ``--force``, ``-f``: Overwrite existing files

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing files"
    )
    return parser.parse_args()


def get_features(namespace):
    """
    Iterate over all Feature classes in a given namespace.

    Parameters
    ----------
    namespace : dict
        A dictionary of name to object.

    Yields
    ------
    feature : Feature
        An instance of a Feature class.

    See Also
    --------
    generate_features
    """
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    """
    Generate features for all classes in a given namespace.

    Parameters
    ----------
    namespace : dict
        A dictionary of name to object.
    overwrite : bool
        If True, overwrite existing feature files.

    Notes
    -----
    This function iterates over all Feature classes in a given namespace,
    checks if the feature files already exist, and runs ``run`` method of
    each feature if the files do not exist or if the overwrite flag is
    set to True.
    """
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, "was skipped")
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    """
    This class defines a template for creating and saving features for machine learning models.
    """

    prefix = ""
    suffix = ""
    dir = "."

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                "([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip("_")

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f"{self.name}_train.feather"
        self.test_path = Path(self.dir) / f"{self.name}_test.feather"

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = "_" + self.suffix if self.suffix else ""
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))
