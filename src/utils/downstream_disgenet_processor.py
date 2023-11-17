import json
import sys
import os
import torch
from tdc.multi_pred import GDA
from utils.data_loader import GDA_Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

sys.path.append("../")

class DisGeNETProcessor:
    def __init__(self, data_dir="/nfs/dpa_pretrain/data/downstream/"):
        train_data = pd.read_csv('/nfs/dpa_pretrain/data/downstream/fold_1/train.csv')
        valid_data = pd.read_csv('/nfs/dpa_pretrain/data/downstream/fold_1/valid.csv')
        valid_data, test_data = train_test_split(valid_data, test_size=1/3, random_state=42)
        # train_data = pd.read_csv('/nfs/dpa_pretrain/data/downstream/fold_1/disgenet_finetune.csv')
        # train_data, valid_data = train_test_split(data, test_size=0.3, random_state=42)
        
        # alzheimer and stomach dataset use [["proteinSeq", "diseaseDes", "Y"]].dropna()
        
        self.name = "DisGeNET"
        self.train_dataset_df = train_data[["proteinSeq", "diseaseDes", "score"]].dropna()
        self.val_dataset_df = valid_data[["proteinSeq", "diseaseDes", "score"]].dropna()
        self.test_dataset_df = test_data[["proteinSeq", "diseaseDes", "score"]].dropna()
        # self.test_dataset_df = test_data[["proteinSeq", "diseaseDes", "Y"]].dropna()

        
    def get_train_examples(self, test=False):
        """get training examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.train_dataset_df["proteinSeq"].values[:4096],
                self.train_dataset_df["diseaseDes"].values[:4096],
                self.train_dataset_df["score"].values[:4096],
            )
        elif test > 1:
            return (
                self.train_dataset_df["proteinSeq"].values[:test],
                self.train_dataset_df["diseaseDes"].values[:test],
                self.train_dataset_df["score"].values[:test],
            )
        else:
            return GDA_Dataset( (
                self.train_dataset_df["proteinSeq"].values,
                self.train_dataset_df["diseaseDes"].values,
                self.train_dataset_df["score"].values,
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
                self.val_dataset_df["proteinSeq"].values[:1024],
                self.val_dataset_df["diseaseDes"].values[:1024],
                self.val_dataset_df["score"].values[:1024],
            )
        elif test > 1:
            return (
                self.val_dataset_df["proteinSeq"].values[:test],
                self.val_dataset_df["diseaseDes"].values[:test],
                self.val_dataset_df["score"].values[:test],
            )
        else:
            return GDA_Dataset((
                self.val_dataset_df["proteinSeq"].values,
                self.val_dataset_df["diseaseDes"].values,
                self.val_dataset_df["score"].values,
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
                 self.test_dataset_df["proteinSeq"].values[:1024],
                 self.test_dataset_df["diseaseDes"].values[:1024],
                 self.test_dataset_df["score"].values[:1024],
             )
        elif test > 1:
             return (
                 self.test_dataset_df["proteinSeq"].values[:test],
                 self.test_dataset_df["diseaseDes"].values[:test],
                 self.test_dataset_df["score"].values[:test],
             )
        else:
             return GDA_Dataset( (
                 self.test_dataset_df["proteinSeq"].values,
                 self.test_dataset_df["diseaseDes"].values,
                 self.test_dataset_df["score"].values,
             ))
