import json
import sys
import os
import torch
from tdc.multi_pred import GDA
from utils.data_loader import GDA_Dataset
import numpy as np
import pandas as pd
sys.path.append("../")


class DisGeNETProcessor:
    def __init__(self, data_dir="nfs/FusionGDA/data/downstream/"):

        data = GDA(name="DisGeNET") # , path=data_dir
        data.neg_sample(frac = 1)
        data.binarize(threshold = 0, order = 'ascending')
        self.datasets = data.get_split(method = 'random', seed = 42, frac = [0.7, 0.1, 0.2])
        self.name = "TDC"
        
        self.train_dataset_df = self.datasets['train']
        self.train_dataset_df = self.train_dataset_df[
            ["Gene", "Disease", "Y"]
        ].dropna() 
        
        self.val_dataset_df = self.datasets["valid"]
        self.val_dataset_df = self.val_dataset_df[
            ["Gene", "Disease", "Y"]
        ].dropna() 
        
        self.test_dataset_df = self.datasets["test"]
        self.test_dataset_df = self.test_dataset_df[
            ["Gene", "Disease", "Y"]
        ].dropna() 
        
    def get_train_examples(self, test=False):
        """get training examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.train_dataset_df["Gene"].values[:4096],
                self.train_dataset_df["Disease"].values[:4096],
                self.train_dataset_df["Y"].values[:4096],
            )
        elif test > 1:
            return (
                self.train_dataset_df["Gene"].values[:test],
                self.train_dataset_df["Disease"].values[:test],
                self.train_dataset_df["Y"].values[:test],
            )
        else:
            return GDA_Dataset( (
                self.train_dataset_df["Gene"].values,
                self.train_dataset_df["Disease"].values,
                self.train_dataset_df["Y"].values,
            ))

    def get_val_examples(self, test=False):
        """get validation examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.val_dataset_df["Gene"].values[:1024],
                self.val_dataset_df["Disease"].values[:1024],
                self.val_dataset_df["Y"].values[:1024],
            )
        elif test > 1:
            return (
                self.val_dataset_df["Gene"].values[:test],
                self.val_dataset_df["Disease"].values[:test],
                self.val_dataset_df["Y"].values[:test],
            )
        else:
            return GDA_Dataset((
                self.val_dataset_df["Gene"].values,
                self.val_dataset_df["Disease"].values,
                self.val_dataset_df["Y"].values,
            ))

    def get_test_examples(self, test=False):
        """get test examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.test_dataset_df["Gene"].values[:1024],
                self.test_dataset_df["Disease"].values[:1024],
                self.test_dataset_df["Y"].values[:1024],
            )
        elif test > 1:
            return (
                self.test_dataset_df["Gene"].values[:test],
                self.test_dataset_df["Disease"].values[:test],
                self.test_dataset_df["Y"].values[:test],
            )
        else:
            return GDA_Dataset( (
                self.test_dataset_df["Gene"].values,
                self.test_dataset_df["Disease"].values,
                self.test_dataset_df["Y"].values,
            ))


