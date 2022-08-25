import argparse
import os
import random
import sys
import time

sys.path.append("../")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from src.utils.data_loader import PPI_Pretrain_Dataset
from src.utils.metric_learning_models import PPI_Metric_Learning

import wandb

wandb.init(project="Aug_pretrain_string_ppi_adapter")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="Rostlab/prot_bert",
        help="path/name of protein encoder model located",
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument("--prot_max_length", type=int, default=512)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--max_step", type=int, default=sys.maxsize)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--reduction_factor", type=int, default=8)
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=float, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/pretrain/ppi_2m_adapter/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", type=str, default="model", help="the name of the saved file"
    )
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def set_random_seed(seed):
    """set random seeds.

    Args:
        seed (int): random seed number.
    """
    # Set random seed for reproducibility
    if seed is None:
        seed = int(time.time())
        print(f"Generated random seed {seed}.")
    else:
        print(f"Received random seed {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def save_model(args, total_step, model):
    if args.use_adapter:
        adapter_save_path = save_dir = os.path.join(
            args.save_path_prefix,
            f"reduction_factor_{args.reduction_factor}",
        )
        model.save_adapters(adapter_save_path, total_step)
        print(f"save model adapters on {args.save_path_prefix}")
    else:
        save_dir = os.path.join(
            args.save_path_prefix,
            f"step_{total_step}_model.bin",
        )
        torch.save(model, save_dir)
        print(f"save model {save_dir} on {args.save_path_prefix}")


def train(args):
    # mixed precision training

    os.makedirs(args.save_path_prefix, exist_ok=True)
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )

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
        query_encodings2 = prot_tokenizer.batch_encode_plus(
            list(query2),
            max_length=args.prot_max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        scores = torch.tensor(list(scores))
        return query_encodings1, query_encodings2, scores

    prot_model = BertModel.from_pretrained(args.prot_encoder_path)
    model = PPI_Metric_Learning(prot_model, args)
    # print(model)
    model.init_adapters(reduction_factor=args.reduction_factor)
    model = model.to(args.device)

    train_data = PPI_Pretrain_Dataset(test=args.test)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_batch_encoding,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    num_train_optimization_steps = (
        int(len(train_data) / args.batch_size / args.gradient_accumulation_steps)
        * args.max_epoch
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        num_training_steps=num_train_optimization_steps,
        num_warmup_steps=args.warmup_steps,
    )

    total_step = 0
    time_start = time.time()

    for epoch in tqdm(range(args.max_epoch), desc="Pre-training"):
        model.train()
        t_loss = 0
        for step, batch in enumerate(train_dataloader):
            prot1_input, prot2_input, label_inputs = batch
            prot1_input = prot1_input.to(args.device)
            prot2_input = prot2_input.to(args.device)
            label_inputs = label_inputs.to(args.device)
            optimizer.zero_grad()

            loss = model(prot1_input, prot2_input, label_inputs)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            t_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                wandb.log({"loss": loss.item()})
                total_step += 1
                if (total_step >= args.save_step) & (total_step % args.save_step == 0):
                    save_model(args, total_step, model)
                    time_end = time.time()
                    print(
                        f"training this {args.save_step} steps cost {(time_start-time_end):.2f} seconds, current total_step:{total_step}"
                    )
                    time_start = time_end
                    if total_step >= args.max_step:
                        break


if __name__ == "__main__":
    args = parse_config()
    if torch.cuda.is_available():
        print("cuda is available.")
        print(f"current device {args}.")
    else:
        args.device = "cpu"

    args.seed = set_random_seed(args.seed)
    wandb.config.update(args)
    train(args)
