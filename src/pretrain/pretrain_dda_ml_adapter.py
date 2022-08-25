import argparse
import os
import sys
import time

sys.path.append("../")

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from src.utils.commons import save_model, set_random_seed
from src.utils.data_loader import DDA_Pretrain_Dataset
from src.utils.metric_learning_models import DDA_Metric_Learning

import wandb

wandb.init(project="Aug_pretrain_disgenet_dda_adapter")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path/name of whole model located",
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument("--disease_max_length", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--max_step", type=int, default=sys.maxsize)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--reduction_factor", type=int, default=8)
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/pretrain/dda_adapter/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", type=str, default="model", help="the name of the saved file"
    )
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def train(args):

    os.makedirs(args.save_path_prefix, exist_ok=True)
    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)

    def collate_fn_batch_encoding(batch):
        query1, query2, scores = zip(*batch)
        query_encodings1 = disease_tokenizer.batch_encode_plus(
            list(query1),
            max_length=args.disease_max_length,
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

    disease_model = BertModel.from_pretrained(args.disease_encoder_path)
    model = DDA_Metric_Learning(disease_model, args)
    # print(model)
    model.init_adapters(reduction_factor=args.reduction_factor)
    model = model.to(args.device)

    train_data = DDA_Pretrain_Dataset(test=args.test)
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
            text1_inputs, text2_inputs, label_inputs = batch
            text1_inputs = text1_inputs.to(args.device)
            text2_inputs = text2_inputs.to(args.device)
            optimizer.zero_grad()

            loss = model(text1_inputs, text2_inputs, label_inputs)
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

