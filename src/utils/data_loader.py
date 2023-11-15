import logging
import sys
import numpy as np
sys.path.append("../")
from tdc.multi_pred import GDA
import pandas as pd
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

class GDA_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL gene-to-disease interactions
    """

    def __init__(self, data_examples):
        self.protein_seqs = data_examples[0]
        self.disease_dess = data_examples[1]
        self.scores = data_examples[2]

    def __getitem__(self, query_idx):

        protein_seq = self.protein_seqs[query_idx]
        disease_des = self.disease_dess[query_idx]
        score = self.scores[query_idx]

        return protein_seq, disease_des, score

    def __len__(self):
        return len(self.protein_seqs)


class GDA_Pretrain_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL gene-disease associations
    """

    def __init__(self, data_dir="../../data/pretrain/", test=False):
        LOGGER.info("Initializing GDA Pretraining Dataset ! ...")
        self.dataset_df = pd.read_csv(f"{data_dir}/disgenet_gda.csv")
        self.dataset_df = self.dataset_df[
            ["proteinSeq", "diseaseDes", "score"]
        ].dropna()  # Drop missing values.
        # print(self.dataset_df.head())
        print(
            f"{data_dir}disgenet_gda.csv loaded, found associations: {len(self.dataset_df.index)}"
        )
        if test:
            self.protein_seqs = self.dataset_df["proteinSeq"].values[:128]
            self.disease_dess = self.dataset_df["diseaseDes"].values[:128]
            self.scores = 128 * [1]
        else:
            self.protein_seqs = self.dataset_df["proteinSeq"].values
            self.disease_dess = self.dataset_df["diseaseDes"].values
            self.scores = len(self.dataset_df["score"].values) * [1]

    def __getitem__(self, query_idx):

        protein_seq = self.protein_seqs[query_idx]
        disease_des = self.disease_dess[query_idx]
        score = self.scores[query_idx]

        return protein_seq, disease_des, score

    def __len__(self):
        return len(self.protein_seqs)