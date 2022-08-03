import os
import json
import numpy as np
import pandas as pd

import csv

import torch

import logging


class GENERIC_WRANGLER(object):
    def __init__(self, folder):
        """Looks at folder, grabs x& y train and test

        Args:
            folder (str): Folder that contains x/y train and test`
        """
        self.folder = folder

        self.load_data()

    def load_data(self):
        def load_from_file(file, folder):
            path = os.path.join(folder, file)
            return pd.read_csv(path)

        self.X_train = load_from_file("x_train.csv", self.folder)
        self.X_test = load_from_file("x_test.csv", self.folder)
        self.y_train = load_from_file("y_train.csv", self.folder)
        self.y_test = load_from_file("y_test.csv", self.folder)

    @property
    def data_lengths(self):
        """xtrain, xtest, ytrain, ytest lengths
        """
        return (
            len(self.X_train),
            len(self.X_test),
            len(self.y_train),
            len(self.y_test),
        )
