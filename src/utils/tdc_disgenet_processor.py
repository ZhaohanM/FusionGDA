import sys

import torch
from tdc.multi_pred import GDA

sys.path.append("../")


class DisGeNETProcessor:
    def __init__(self, data_dir="../data/downstream/"):
        data = GDA(name="DisGeNET", path=data_dir)
        self.datasets = data.get_split()
        self.name = "DisGeNET"

    def get_train_examples(self, test=False):
        """get training examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.datasets["train"]["Gene"].values[:4096],
                self.datasets["train"]["Disease"].values[:4096],
                self.datasets["train"]["Y"].values[:4096],
            )
        elif test > 1:
            return (
                self.datasets["train"]["Gene"].values[:test],
                self.datasets["train"]["Disease"].values[:test],
                self.datasets["train"]["Y"].values[:test],
            )
        else:
            return (
                self.datasets["train"]["Gene"].values,
                self.datasets["train"]["Disease"].values,
                self.datasets["train"]["Y"].values,
            )

    def get_dev_examples(self, test=False):
        """get validation examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.datasets["valid"]["Gene"].values[:1024],
                self.datasets["valid"]["Disease"].values[:1024],
                self.datasets["valid"]["Y"].values[:1024],
            )
        elif test > 1:
            return (
                self.datasets["valid"]["Gene"].values[:test],
                self.datasets["valid"]["Disease"].values[:test],
                self.datasets["valid"]["Y"].values[:test],
            )
        else:
            return (
                self.datasets["valid"]["Gene"].values,
                self.datasets["valid"]["Disease"].values,
                self.datasets["valid"]["Y"].values,
            )

    def get_test_examples(self, test=False):
        """get test examples

        Args:
            test (bool, optional): test can be int or bool. If test>1, will take test as the number of test examples. Defaults to False.

        Returns:
            _type_: _description_
        """
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.datasets["test"]["Gene"].values[:1024],
                self.datasets["test"]["Disease"].values[:1024],
                self.datasets["test"]["Y"].values[:1024],
            )
        elif test > 1:
            return (
                self.datasets["test"]["Gene"].values[:test],
                self.datasets["test"]["Disease"].values[:test],
                self.datasets["test"]["Y"].values[:test],
            )
        else:
            return (
                self.datasets["test"]["Gene"].values,
                self.datasets["test"]["Disease"].values,
                self.datasets["test"]["Y"].values,
            )


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
