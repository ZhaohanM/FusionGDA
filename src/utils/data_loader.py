import logging
import sys

sys.path.append("../")

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


class PPI_Pretrain_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL protein-to-protein interactions
    """

    def __init__(self, data_dir="../../data/pretrain/", test=False):
        LOGGER.info("Initializing metric learning data set! ...")
        self.dataset_df = pd.read_csv(f"{data_dir}/string_ppi_900_2m.csv")
        self.dataset_df = self.dataset_df[["item_seq_a", "item_seq_b", "score"]]
        self.dataset_df = self.dataset_df.dropna()
        if test:
            self.dataset_df = self.dataset_df.sample(100)
        print(
            f"{data_dir}/string_ppi_900_2m.csv loaded, found interactions: {len(self.dataset_df.index)}"
        )
        self.protein_seq1 = self.dataset_df["item_seq_a"].values
        self.protein_seq2 = self.dataset_df["item_seq_b"].values
        self.scores = len(self.dataset_df["score"].values) * [1]

    def __getitem__(self, query_idx):

        protein_seq1 = self.protein_seq1[query_idx]
        protein_seq2 = self.protein_seq2[query_idx]
        score = self.scores[query_idx]

        return protein_seq1, protein_seq2, score

    def __len__(self):
        return len(self.protein_seq1)


class PPI_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL protein-to-protein interactions
    """

    def __init__(self, protein_seq1, protein_seq2, score):
        self.protein_seq1 = protein_seq1
        self.protein_seq2 = protein_seq2
        self.scores = score

    def __getitem__(self, query_idx):

        protein_seq1 = self.protein_seq1[query_idx]
        protein_seq2 = self.protein_seq2[query_idx]
        score = self.scores[query_idx]

        return protein_seq1, protein_seq2, score

    def __len__(self):
        return len(self.protein_seq1)


class DDA_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL disease-to-disease associations
    """

    def __init__(self, diseaseDes1, diseaseDes2, label):
        self.diseaseDes1 = diseaseDes1
        self.diseaseDes2 = diseaseDes2
        self.label = label

    def __getitem__(self, query_idx):

        diseaseDes1 = self.diseaseDes1[query_idx]
        diseaseDes2 = self.diseaseDes2[query_idx]
        label = self.label[query_idx]

        return diseaseDes1, diseaseDes2, label

    def __len__(self):
        return len(self.diseaseDes1)


class DDA_Pretrain_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL protein-to-protein interactions
    """
    print(Dataset)
    def __init__(self, data_dir="../../data/pretrain/", test=False):
        LOGGER.info("Initializing metric learning data set! ...")
        self.dataset_df = pd.read_csv(f"{data_dir}disgenet_dda.csv")
        self.dataset_df = self.dataset_df.dropna()  # Drop missing values.
        if test:
            self.dataset_df = self.dataset_df.sample(100)
        print(
            f"{data_dir}disgenet_dda.csv loaded, found associations: {len(self.dataset_df.index)}"
        )
        self.disease_des1 = self.dataset_df["diseaseDes1"].values
        self.disease_des2 = self.dataset_df["diseaseDes2"].values
        self.scores = len(self.dataset_df["jaccard_variant"].values) * [1]

    def __getitem__(self, query_idx):

        disease_des1 = self.disease_des1[query_idx]
        disease_des2 = self.disease_des2[query_idx]
        score = self.scores[query_idx]

        return disease_des1, disease_des2, score

    def __len__(self):
        return len(self.disease_des1)
