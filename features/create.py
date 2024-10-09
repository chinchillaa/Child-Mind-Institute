import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = "features"


class Sex(Feature):
    def create_features(self):
        self.train["Sex"] = train["Sex"].replace(["male", "female"], [0, 1])
        self.test["Sex"] = test["Sex"].replace(["male", "female"], [0, 1])


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_parquet("./data/input/train_preprocessed.parquet")
    test = pd.read_parquet("./data/input/test_preprocessed.parquet")

    generate_features(globals(), args.force)
