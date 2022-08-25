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
from src.utils.string_ppi_processor import (
    STRINGPPIProcessor,
    convert_ppi_examples_to_tokens,
    convert_ppi_tokens_to_tensors,
)
from src.utils.tdc_disgenet_processor import DisGeNETProcessor as DisGeNETProcessor_ft
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.models import GDANet_model2

import wandb

wandb.init(project="july_pretrain_disgenet_dda")


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
        "--ppi_dir",
        type=str,
        default=None,
        help="path of training data",
    )
    parser.add_argument(
        "--train_ppi_data_save_name",
        type=str,
        default="string_ppi.pt",
        help="path of tokenized training data",
    )
    parser.add_argument(
        "--train_dpa_data_save_name",
        type=str,
        default="disgenet_dpa.pt",
        help="path of tokenized training data",
    )
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/pretrain/model2/",
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
        model = GDANet_model2(
            prot_model,
            disease_model,
            freeze_prot_encoder=True,
            freeze_disease_encoder=False,
        ).to(args.device)
        prior_epoch = -1  # No prior epoch and step
        prior_step = -1
    print(model)
    model.freeze_encoders(args.freeze_prot_encoder, args.freeze_disease_encoder)
    model.train()

    if args.ppi_dir is not None:
        args.train_ppi_data_save_name = args.ppi_dir.split("/")[-1][:-4] + ".pt"
    train_ppi_data_save_path = os.path.join(
        args.train_dir, args.train_ppi_data_save_name
    )
    if os.path.exists(train_ppi_data_save_path) and args.test == False:
        print(f"load prior train_data from {train_ppi_data_save_path}.")
        ppi_train_data = torch.load(train_ppi_data_save_path)
    else:
        print("loading STRING ppi dataset.")
        STRINGPPI = STRINGPPIProcessor(data_dir=args.ppi_dir)
        ppi_examples = STRINGPPI.get_train_examples(test=args.test)
        ppi_tokens = convert_ppi_examples_to_tokens(args, ppi_examples, prot_tokenizer)
        ppi_inputs = convert_ppi_tokens_to_tensors(ppi_tokens, "cpu")
        ppi_train_data = TensorDataset(
            ppi_inputs["prot1_inputs"],
            ppi_inputs["prot2_inputs"],
            ppi_inputs["label_inputs"],
        )
        print(f"save train_ppi_data into {train_ppi_data_save_path}.")
        torch.save(ppi_train_data, train_ppi_data_save_path)
    ppi_train_sampler = RandomSampler(ppi_train_data)
    ppi_train_dataloader = DataLoader(
        ppi_train_data,
        sampler=ppi_train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_dpa_data_save_path = os.path.join(
        args.train_dir, args.train_dpa_data_save_name
    )
    if os.path.exists(train_dpa_data_save_path) and args.test == False:
        print(f"load prior train_dpa_data from {train_dpa_data_save_path}.")
        dpa_train_data = torch.load(train_dpa_data_save_path)
    else:
        print("loading DisGeNET dpa dataset.")
        disGeNET = DisGeNETProcessor()
        dpa_examples = disGeNET.get_train_examples(test=args.test)
        dpa_tokens = convert_examples_to_tokens(
            args, dpa_examples, prot_tokenizer, disease_tokenizer
        )
        dpa_inputs = convert_tokens_to_tensors(dpa_tokens, "cpu")
        dpa_train_data = TensorDataset(
            dpa_inputs["prot_inputs"],
            dpa_inputs["disease_inputs"],
            dpa_inputs["label_inputs"],
        )
        print(f"save train_dpa_data into {train_dpa_data_save_path}.")
        torch.save(dpa_train_data, train_dpa_data_save_path)
    dpa_train_sampler = RandomSampler(dpa_train_data)
    dpa_train_dataloader = DataLoader(
        dpa_train_data,
        sampler=dpa_train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Loading valid dataset for monitor pretraining
    print("loading DisGeNET finetuning dataset.")
    disGeNET_ft = DisGeNETProcessor_ft()
    dev_examples = disGeNET_ft.get_dev_examples(test=args.test)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    dpa_loss = nn.MSELoss()
    ppi_loss = nn.MSELoss()
    metric = nn.MSELoss(reduction="sum")
    max_step = max(len(dpa_train_dataloader), len(ppi_train_dataloader))
    total_step = 0
    dpa_train_iter = iter(dpa_train_dataloader)
    ppi_train_iter = iter(ppi_train_dataloader)
    for epoch in tqdm(range(10), desc="Training"):
        if prior_epoch != -1 and epoch < prior_epoch:
            continue
        model.train()
        t_loss = 0
        for step in tqdm(range(max_step)):
            if prior_step != -1 and step < prior_step:
                continue
            # get dpa batch data
            next_dba_batch = next(dpa_train_iter, None)
            if next_dba_batch is not None:
                dpa_prot_input, dpa_disease_inputs, dpa_label_inputs = next_dba_batch
            else:
                dpa_train_iter = iter(dpa_train_dataloader)
                next_dba_batch = next(dpa_train_iter, None)
                dpa_prot_input, dpa_disease_inputs, dpa_label_inputs = next_dba_batch
            # get ppi batch data
            next_ppi_batch = next(ppi_train_iter, None)
            if next_ppi_batch is not None:
                ppi_prot1_input, ppi_prot2_input, ppi_label_inputs = next_ppi_batch
            else:
                ppi_train_iter = iter(ppi_train_dataloader)
                next_ppi_batch = next(ppi_train_iter, None)
                ppi_prot1_input, ppi_prot2_input, ppi_label_inputs = next_ppi_batch

            dpa_prot_input = dpa_prot_input.to(args.device)
            dpa_disease_inputs = dpa_disease_inputs.to(args.device)
            dpa_label_inputs = dpa_label_inputs.to(args.device)
            ppi_prot1_input = ppi_prot1_input.to(args.device)
            ppi_prot2_input = ppi_prot2_input.to(args.device)
            ppi_label_inputs = ppi_label_inputs.to(args.device)
            optimizer.zero_grad()

            dpa_out = model(dpa_prot_input, dpa_disease_inputs)
            ppi_out = model.forward_ppi(ppi_prot1_input, ppi_prot2_input)
            dpa_output = dpa_loss(dpa_out, dpa_label_inputs)
            ppi_output = ppi_loss(ppi_out, ppi_label_inputs)
            output = dpa_output + (args.alpha * ppi_output)
            t_loss += output.item() + (args.alpha * output.item())
            output.backward()
            optimizer.step()
            total_step += 1
            if (total_step >= args.save_step) & (total_step % args.save_step == 0):
                save_dir = os.path.join(
                    args.save_path_prefix,
                    f"model2_step_{total_step}.bin",
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
