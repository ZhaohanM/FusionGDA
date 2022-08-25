import sys

import pandas as pd

sys.path.append("../")

import sys

import torch

sys.path.append("../")


class DDAProcessor:
    def __init__(self, data_dir="../data/pretrain/"):
        dda_data_path = f"{data_dir}disgenet_gda.csv"
        self.data_name = dda_data_path.split("/")[-1][:-4]
        self.dataset_df = pd.read_csv(dda_data_path)
        print(f"{dda_data_path} loaded")

    def get_train_examples(self, test=False):
        if test:
            return (
                self.dataset_df["diseaseDes1"].values[:512],
                self.dataset_df["diseaseDes2"].values[:512],
                self.dataset_df["jaccard_variant"].values[:512],
            )
        return (
            self.dataset_df["diseaseDes1"].values,
            self.dataset_df["diseaseDes2"].values,
            self.dataset_df["jaccard_variant"].values,
        )


def convert_dda_examples_to_tokens(
    args, examples, prot_tokenizer, max_seq_length=512, test=False
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
    for prot1, prot2, label in zip(*examples):
        first_sentences.append([" ".join(prot1)])
        second_sentences.append([" ".join(prot2)])
        labels.append([label])

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    if test:
        first_sentences = first_sentences[:1024]
        second_sentences = second_sentences[:1024]
        labels = labels[:1024]

    print("start tokenizing ...")
    # Tokenize
    prot1_tokens = prot_tokenizer(
        first_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )["input_ids"]
    # Tokenize
    prot2_tokens = prot_tokenizer(
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )["input_ids"]

    print("finish tokenizing ...")

    inputs = {}
    inputs["prot1_tokens"] = prot1_tokens
    inputs["prot2_tokens"] = prot2_tokens
    inputs["labels"] = labels
    return inputs


def convert_dda_tokens_to_tensors(tokens, device="cpu"):
    input_dict = {}
    prot1_inputs = torch.tensor(tokens["prot1_tokens"], dtype=torch.long, device=device)
    prot2_inputs = torch.tensor(tokens["prot2_tokens"], dtype=torch.long, device=device)
    labels_inputs = torch.tensor(tokens["labels"], dtype=torch.float, device=device)
    input_dict["prot1_inputs"] = prot1_inputs
    input_dict["prot2_inputs"] = prot2_inputs
    input_dict["label_inputs"] = labels_inputs
    return input_dict


# Test
# disGeNET = DisGeNETProcessor()
# examples = disGeNET.get_train_examples()
# tokens = convert_examples_to_tokens(examples, prot_tokenizer, disease_tokenizer, max_seq_length=5,test=True)
# inputs = convert_tokens_to_tensors(tokens, device)
