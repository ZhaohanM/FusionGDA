import argparse
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from src.utils.tdc_disgenet_processor import (
    DisGeNETProcessor,
    convert_examples_to_tokens,
    convert_tokens_to_tensors,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.models import GDANet

import wandb


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="Rostlab/prot_bert",
        help="path/name of protein encoder model located",
    )
    parser.add_argument(
        "--disease_encoder_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="path/name of textual pre-trained language model",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="../data/",
        help="path of training data",
    )
    parser.add_argument("--freeze_prot_encoder", type=bool, default=False)
    parser.add_argument("--freeze_disease_encoder", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--save_path_prefix", type=str, help="save the result in which directory"
    )
    parser.add_argument("--save_name", type=str, help="the name of the saved file")
    return parser.parse_args()


def train_an_epoch(model, train_dataloader, optimizer, loss):
    model.train()
    t_loss = 0
    for step, batch in enumerate(train_dataloader):
        prot_input, disease_inputs, label_inputs = batch
        optimizer.zero_grad()
        out = model(prot_input, disease_inputs)
        output = loss(out, label_inputs)
        t_loss += output.item()
        output.backward()
        optimizer.step()
    return t_loss


def evaluate(model, test_dataloader, optimizer, metric):
    model.eval()
    metric_val = 0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            prot_input, disease_inputs, label_inputs = batch
            optimizer.zero_grad()
            out = model(prot_input, disease_inputs)
            output = metric(out, label_inputs)
            metric_val += output.item()
    return metric_val / (step + 1)


def train(args):
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )
    prot_model = BertModel.from_pretrained(args.prot_encoder_path)
    prot_model = prot_model.to(args.device)

    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)
    disease_model = BertModel.from_pretrained(args.disease_encoder_path)
    disease_model = disease_model.to(args.device)

    disGeNET = DisGeNETProcessor()
    examples = disGeNET.get_train_examples()
    tokens = convert_examples_to_tokens(
        args, examples, prot_tokenizer, disease_tokenizer
    )
    inputs = convert_tokens_to_tensors(tokens, args.device)
    train_data = TensorDataset(
        inputs["prot_input"], inputs["disease_inputs"], inputs["label_inputs"]
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size
    )

    model = GDANet(
        prot_model,
        disease_model,
        freeze_prot_encoder=True,
        freeze_disease_encoder=False,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    loss = nn.MSELoss()
    for epoch in tqdm(range(200), desc="Training"):
        model.train()
        # epoch_loss = train_an_epoch(model, train_dataloader, optimizer, loss)
        t_loss = 0
        for step, batch in enumerate(train_dataloader):
            prot_input, disease_inputs, label_inputs = batch
            optimizer.zero_grad()
            out = model(prot_input, disease_inputs)
            output = loss(out, label_inputs)
            t_loss += output.item()
            output.backward()
            optimizer.step()
        print(t_loss)


if __name__ == "__main__":
    args = parse_config()
    if torch.cuda.is_available():
        print("cuda is available.")
        print(f"current device {args}.")
    else:
        args.device = "cpu"
    train(args)
