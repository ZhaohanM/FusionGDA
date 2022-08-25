import argparse
import os
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from src.utils.disgenet_gda_processor import (
    DisGeNETProcessor,
    convert_examples_to_tokens,
    convert_tokens_to_tensors,
)
from src.utils.tdc_disgenet_processor import DisGeNETProcessor as DisGeNETProcessor_ft
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.models import GDANet

import wandb

wandb.init(project="july_pretrain_disgenet_22_may")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path/name of whole model located",
    )
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
        default="../data/pretrain/",
        help="path of training data",
    )
    parser.add_argument(
        "--train_data_save_name",
        type=str,
        default="disgenet.pt",
        help="path of tokenized training data",
    )
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/pretrain/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", type=str, default="model", help="the name of the saved file"
    )
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
    count = 0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            prot_input, disease_inputs, label_inputs = batch
            optimizer.zero_grad()
            out = model(prot_input, disease_inputs)
            output = metric(out, label_inputs)
            metric_val += output.item()
            count += label_inputs.size()[0]
    return round(metric_val / count, 4)


def train(args):
    os.makedirs(args.save_path_prefix, exist_ok=True)
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )
    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)
    if args.model_path:
        model = torch.load(args.model_path, map_location=torch.device(args.device))
        prior_epoch = int(args.model_path.split("epoch_")[1].split("_step_")[0]) - 1
        prior_step = int(
            args.model_path.split("epoch_")[1].split("_step_")[1].replace(".bin", "")
        )  # model_path should be "xxx/xxx/epoch_xxx_step_xxx.bin"
        print(f"load prior model {args.model_path}.")
    else:
        prot_model = BertModel.from_pretrained(args.prot_encoder_path)
        prot_model = prot_model.to(args.device)
        disease_model = BertModel.from_pretrained(args.disease_encoder_path)
        disease_model = disease_model.to(args.device)
        model = GDANet(
            prot_model,
            disease_model,
            freeze_prot_encoder=True,
            freeze_disease_encoder=False,
        ).to(args.device)
        prior_epoch = -1  # No prior epoch and step
        prior_step = -1
    # print(model)
    model.freeze_encoders(args.freeze_prot_encoder, args.freeze_disease_encoder)
    model.train()
    train_data_save_path = os.path.join(args.train_dir, args.train_data_save_name)
    if os.path.exists(train_data_save_path):
        print(f"load prior train_data from {train_data_save_path}.")
        train_data = torch.load(train_data_save_path)
    else:
        print("loading dataset.")
        disGeNET = DisGeNETProcessor()
        examples = disGeNET.get_train_examples()
        tokens = convert_examples_to_tokens(
            args, examples, prot_tokenizer, disease_tokenizer, test=args.test
        )
        inputs = convert_tokens_to_tensors(tokens, "cpu")
        train_data = TensorDataset(
            inputs["prot_inputs"], inputs["disease_inputs"], inputs["label_inputs"]
        )
        print(f"save train_data into {train_data_save_path}.")
        torch.save(train_data, train_data_save_path)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Loading valid dataset for monitor pretraining
    disGeNET_ft = DisGeNETProcessor_ft()
    dev_examples = disGeNET_ft.get_dev_examples()
    dev_tokens = convert_examples_to_tokens(
        args, dev_examples, prot_tokenizer, disease_tokenizer
    )
    dev_inputs = convert_tokens_to_tensors(dev_tokens, args.device)
    dev_data = TensorDataset(
        dev_inputs["prot_inputs"],
        dev_inputs["disease_inputs"],
        dev_inputs["label_inputs"],
    )
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data, sampler=dev_sampler, batch_size=args.batch_size
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss = nn.MSELoss()
    metric = nn.MSELoss(reduction="sum")
    total_step = 0
    for epoch in tqdm(range(10), desc="Training"):
        if prior_epoch != -1 and epoch < prior_epoch:
            continue
        model.train()
        t_loss = 0
        for step, batch in enumerate(train_dataloader):
            if prior_step != -1 and step < prior_step:
                continue
            prot_input, disease_inputs, label_inputs = batch
            prot_input = prot_input.to(args.device)
            disease_inputs = disease_inputs.to(args.device)
            label_inputs = label_inputs.to(args.device)
            optimizer.zero_grad()
            out = model(prot_input, disease_inputs)
            output = loss(out, label_inputs)
            t_loss += output.item()
            output.backward()
            optimizer.step()
            total_step += 1
            if (total_step >= args.save_step) & (total_step % args.save_step == 0):
                save_dir = os.path.join(
                    args.save_path_prefix,
                    f"step_{total_step}_model.bin",
                )
                torch.save(model, save_dir)
                print(f"save model {save_dir} on {args.save_path_prefix}")
                mse = evaluate(model, dev_dataloader, optimizer, metric)
                wandb.log({"total_step": total_step, "dev_mse": mse})


if __name__ == "__main__":
    args = parse_config()
    if torch.cuda.is_available():
        print("cuda is available.")
        print(f"current device {args}.")
    else:
        args.device = "cpu"
    wandb.config.update(args)
    train(args)
