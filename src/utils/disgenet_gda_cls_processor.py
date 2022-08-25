import json
import sys

sys.path.append("../")

import numpy as np
import pandas as pd
import torch

from src.utils.data_loader import GDA_Dataset


class DisGeNETProcessor:
    def __init__(self, data_dir="../../data/downstream/"):
        self.train_dataset_df = pd.read_csv(f"{data_dir}disgenet_gda_cls_train.csv")
        print(
            f"{data_dir}disgenet_gda_cls_train.csv loaded. Total: {len(self.train_dataset_df.index)}"
        )
        self.val_dataset_df = (
            pd.read_csv(f"{data_dir}disgenet_gda_cls_val.csv")
            .sample(frac=1)
            .reset_index()
        )
        print(
            f"{data_dir}disgenet_gda_cls_val.csv loaded. Total: {len(self.val_dataset_df.index)}"
        )
        self.test_dataset_df = (
            pd.read_csv(f"{data_dir}disgenet_gda_cls_test.csv")
            .sample(frac=1)
            .reset_index()
        )
        print(
            f"{data_dir}disgenet_gda_cls_test.csv loaded. Total: {len(self.test_dataset_df.index)}"
        )

        self.max_protein_feat_id = 0
        self.max_disease_feat_id = 0
        for _, item in self.train_dataset_df.iterrows():
            protein_k, _ = zip(*(json.loads(item["protein_feature"]).items()))
            max_protein_id_c = max([int(i) for i in list(protein_k)])
            self.max_protein_feat_id = max(self.max_protein_feat_id, max_protein_id_c)

            disease_k, _ = zip(*(json.loads(item["disease_feature"]).items()))
            max_disease_id_c = max([int(i) for i in list(disease_k)])
            self.max_disease_feat_id = max(self.max_disease_feat_id, max_disease_id_c)
        print(
            f"max_disease_feat_id:{self.max_disease_feat_id}, max_protein_feat_id:{self.max_protein_feat_id}"
        )

    def get_train_examples(self, test=False):
        if test:
            return GDA_Dataset(
                (
                    self.train_dataset_df["proteinSeq"].values[:256],
                    self.train_dataset_df["diseaseDes"].values[:256],
                    self.train_dataset_df["label"].values[:256],
                )
            )
        return GDA_Dataset(
            (
                self.train_dataset_df["proteinSeq"].values,
                self.train_dataset_df["diseaseDes"].values,
                self.train_dataset_df["label"].values,
            )
        )

    def get_dev_examples(self, test=False):
        if test:
            return GDA_Dataset(
                (
                    self.train_dataset_df["proteinSeq"].values[:256],
                    self.train_dataset_df["diseaseDes"].values[:256],
                    self.train_dataset_df["label"].values[:256],
                )
            )
        return GDA_Dataset(
            (
                self.val_dataset_df["proteinSeq"].values,
                self.val_dataset_df["diseaseDes"].values,
                self.val_dataset_df["label"].values,
            )
        )

    def get_test_examples(self, test=False):
        if test:
            return GDA_Dataset(
                (
                    self.train_dataset_df["proteinSeq"].values[:256],
                    self.train_dataset_df["diseaseDes"].values[:256],
                    self.train_dataset_df["label"].values[:256],
                )
            )
        return GDA_Dataset(
            (
                self.test_dataset_df["proteinSeq"].values,
                self.test_dataset_df["diseaseDes"].values,
                self.test_dataset_df["label"].values,
            )
        )

    def get_train_v1_feature(self):
        return self.parse_np_feat(self.train_dataset_df)

    def get_valid_v1_feature(self):
        return self.parse_np_feat(self.val_dataset_df)

    def get_test_v1_feature(self):
        return self.parse_np_feat(self.test_dataset_df)

    def parse_np_feat(self, dataset_df):
        protein_feature = dataset_df["protein_feature"].values
        disease_feature = dataset_df["disease_feature"].values
        labels = dataset_df["label"].values
        # max feature id: 43932
        protein_feat_list = list()
        disease_feat_list = list()
        for prot_str, dis_str in zip(protein_feature, disease_feature):
            protein_feat = np.zeros(self.max_protein_feat_id + 1)
            prot_k, prot_v = zip(*(json.loads(prot_str).items()))
            prot_k = [int(i) for i in list(prot_k)]
            protein_feat[prot_k] = prot_v
            protein_feat_list.append(protein_feat)

            disease_feat = np.zeros(self.max_disease_feat_id + 1)
            dis_k, dis_v = zip(*(json.loads(dis_str).items()))
            dis_k = [int(i) for i in list(dis_k)]
            disease_feat[dis_k] = dis_v
            disease_feat_list.append(disease_feat)

        protein_feat_np = np.stack(protein_feat_list)
        disease_feat_np = np.stack(disease_feat_list)
        return np.hstack((protein_feat_np, disease_feat_np)), labels


def convert_examples_to_tokens(
    args, examples, prot_tokenizer, disease_tokenizer, max_seq_length=512
):
    """Convert both protein and disease examples to token ids

    Args:
        examples (tuple): (Gene(ndarray),Disease(ndarray),Y(ndarray))
        prot_tokenizer (_type_): ProtBERT tokenizer
        disease_tokenizer (_type_): BERT tokenizer
        max_seq_length (int, optional): max_seq_length. Defaults to 512.
        test (bool, optional): use test mode, then only 1024 example will be used for training, validating and testing. Defaults to False.

    Returns:
        _type_: _description_
    """

    first_sentences = []
    second_sentences = []
    labels = []
    for gene, disease, label in zip(*examples):
        first_sentences.append([" ".join(gene)])
        second_sentences.append([disease])
        labels.append([label])

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    print("start tokenizing ...")
    # Tokenize
    prot_tokens = prot_tokenizer(
        first_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )["input_ids"]
    # Tokenize
    disease_tokens = disease_tokenizer(
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )["input_ids"]

    print("finish tokenizing ...")

    inputs = {}
    inputs["prot_tokens"] = prot_tokens
    inputs["disease_tokens"] = disease_tokens
    inputs["labels"] = labels
    return inputs


def convert_tokens_to_tensors(tokens, device="cpu"):
    input_dict = {}
    prot_inputs = torch.tensor(tokens["prot_tokens"], dtype=torch.long, device=device)
    disease_inputs = torch.tensor(
        tokens["disease_tokens"], dtype=torch.long, device=device
    )
    labels_inputs = torch.tensor(tokens["labels"], dtype=torch.float, device=device)
    input_dict["prot_inputs"] = prot_inputs
    input_dict["disease_inputs"] = disease_inputs
    input_dict["label_inputs"] = labels_inputs
    return input_dict


# Test
# disGeNET = DisGeNETProcessor()
# examples = disGeNET.get_train_examples()
# tokens = convert_examples_to_tokens(examples, prot_tokenizer, disease_tokenizer, max_seq_length=5,test=True)
# inputs = convert_tokens_to_tensors(tokens, device)
