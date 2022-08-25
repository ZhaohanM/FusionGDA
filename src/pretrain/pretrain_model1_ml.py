import argparse
import os
import sys
import time

sys.path.append("../")

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.data_loader import GDA_Pretrain_Dataset
from src.utils.metric_learning_models import GD_Metric_Learning

import wandb

wandb.init(project="july_pretrain_disgenet_adapter")


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
        "--amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument(
        "--loss",
        help="{ms_loss|infoNCE|cosine_loss|circle_loss|triplet_loss}}",
        default="infoNCE",
    )
    parser.add_argument(
        "--agg_mode", default="cls", type=str, help="{cls|mean|mean_all_tok}"
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
        default="model1_adapter_disgenet.pt",
        help="path of tokenized training data",
    )
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prot_max_length", type=int, default=512)
    parser.add_argument("--disease_max_length", type=int, default=512)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
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


def train(args):
    # mixed precision training
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    os.makedirs(args.save_path_prefix, exist_ok=True)
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )
    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)

    def collate_fn_batch_encoding(batch):
        query1, query2, scores = zip(*batch)
        query_encodings1 = prot_tokenizer.batch_encode_plus(
            list(query1),
            max_length=args.prot_max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        query_encodings2 = disease_tokenizer.batch_encode_plus(
            list(query2),
            max_length=args.disease_max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        scores = torch.tensor(list(scores))
        return query_encodings1, query_encodings2, scores

    prot_model = BertModel.from_pretrained(args.prot_encoder_path)
    prot_model = prot_model.to(args.device)
    disease_model = BertModel.from_pretrained(args.disease_encoder_path)
    disease_model = disease_model.to(args.device)
    model = GD_Metric_Learning(prot_model, disease_model, 1024, 768, args).to(
        args.device
    )
    # print(model)
    model.freeze_encoders(args.freeze_prot_encoder, args.freeze_disease_encoder)

    train_data = GDA_Pretrain_Dataset()
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_batch_encoding,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_step = 0
    time_start = time.time()

    for epoch in tqdm(range(args.max_epoch), desc="Pre-training"):
        model.train()
        t_loss = 0
        for step, batch in enumerate(train_dataloader):
            prot_input, disease_inputs, label_inputs = batch
            prot_input = prot_input.to(args.device)
            disease_inputs = disease_inputs.to(args.device)
            label_inputs = label_inputs.to(args.device)
            optimizer.zero_grad()

            if args.amp:
                with autocast():
                    loss = model(prot_input, disease_inputs, label_inputs)
            else:
                loss = model(prot_input, disease_inputs, label_inputs)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            t_loss += loss.item()
            optimizer.step()
            total_step += 1
            if (total_step >= args.save_step) & (total_step % args.save_step == 0):
                save_dir = os.path.join(
                    args.save_path_prefix,
                    f"step_{total_step}_model.bin",
                )
                torch.save(model, save_dir)
                print(f"save model {save_dir} on {args.save_path_prefix}")
                time_end = time.time()
                print(
                    f"training this {args.save_step} steps cost {(time_start-time_end):.2f} seconds, current total_step:{total_step}"
                )
                time_start = time_end
                wandb.log({"total_step": total_step, "total_loss": t_loss})


if __name__ == "__main__":
    args = parse_config()
    if torch.cuda.is_available():
        print("cuda is available.")
        print(f"current device {args}.")
    else:
        args.device = "cpu"
    wandb.config.update(args)
    train(args)
