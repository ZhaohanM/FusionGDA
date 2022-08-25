import sys

sys.path.append("../")

import pandas as pd
import torch


class DisGeNETProcessor:
    def __init__(self, data_dir="../data/pretrain/"):
        self.dataset_df = pd.read_csv(f"{data_dir}/disgenet_gda.csv")
        print(f"{data_dir}/disgenet_all.csv loaded")

    def get_train_examples(self, test=False):
        if test:
            return (
                self.dataset_df["proteinSeq"].values[:512],
                self.dataset_df["diseaseDes"].values[:512],
                self.dataset_df["score"].values[:512],
            )
        return (
            self.dataset_df["proteinSeq"].values,
            self.dataset_df["diseaseDes"].values,
            self.dataset_df["score"].values,
        )


def convert_examples_to_tokens(
    args, examples, prot_tokenizer, disease_tokenizer, max_seq_length=512, test=False
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
    if test:
        first_sentences = first_sentences[:1024]
        second_sentences = second_sentences[:1024]
        labels = labels[:1024]

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
