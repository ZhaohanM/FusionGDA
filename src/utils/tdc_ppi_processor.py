import sys

import pandas as pd

sys.path.append("../")
from src.utils.data_loader import PPI_Dataset


class TDCPPIProcessor:
    def __init__(self, data_dir="../../data/downstream"):
        self.train_dataset_df = pd.read_csv(f"{data_dir}/tdc_ppi_6k_train.csv")
        print(
            f"{data_dir}/tdc_ppi_6k_train.csv loaded. Total: {len(self.train_dataset_df.index)}"
        )
        self.val_dataset_df = (
            pd.read_csv(f"{data_dir}/tdc_ppi_6k_valid.csv").sample(frac=1).reset_index()
        )
        print(
            f"{data_dir}/tdc_ppi_6k_valid.csv loaded. Total: {len(self.val_dataset_df.index)}"
        )
        self.test_dataset_df = (
            pd.read_csv(f"{data_dir}/tdc_ppi_6k_test.csv").sample(frac=1).reset_index()
        )
        print(
            f"{data_dir}/tdc_ppi_6k_test.csv loaded. Total: {len(self.test_dataset_df.index)}"
        )

    def get_train_examples(self, test=False):
        if test:
            return PPI_Dataset(
                self.train_dataset_df["Protein1"].values[:100],
                self.train_dataset_df["Protein2"].values[:100],
                self.train_dataset_df["Y"].values[:100],
            )
        return PPI_Dataset(
            self.train_dataset_df["Protein1"].values,
            self.train_dataset_df["Protein2"].values,
            self.train_dataset_df["Y"].values,
        )

    def get_dev_examples(self, test=False):
        if test:
            return PPI_Dataset(
                self.train_dataset_df["Protein1"].values[:100],
                self.train_dataset_df["Protein2"].values[:100],
                self.train_dataset_df["Y"].values[:100],
            )
        return PPI_Dataset(
            self.val_dataset_df["Protein1"].values,
            self.val_dataset_df["Protein2"].values,
            self.val_dataset_df["Y"].values,
        )

    def get_test_examples(self, test=False):
        if test:
            return PPI_Dataset(
                self.train_dataset_df["Protein1"].values[:100],
                self.train_dataset_df["Protein2"].values[:100],
                self.train_dataset_df["Y"].values[:100],
            )
        return PPI_Dataset(
            self.test_dataset_df["Protein1"].values,
            self.test_dataset_df["Protein2"].values,
            self.test_dataset_df["Y"].values,
        )
