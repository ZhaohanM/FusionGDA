import argparse
import os
import random
import string
import sys
from datetime import datetime

sys.path.append("../")

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.disgenet_gda_cls_processor import DisGeNETProcessor
from src.utils.metric_learning_models import GDA_Metric_Learning

import wandb

wandb.init(project="july_finetune_gda_cls_with_adapter")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path/name of the whole model",
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
        default="../data/",
        help="path of training data",
    )
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=8)
    parser.add_argument(
        "--loss",
        help="{ms_loss|infoNCE|cosine_loss|circle_loss|triplet_loss}}",
        default="infoNCE",
    )
    parser.add_argument(
        "--agg_mode", default="cls", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epoch", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/finetune/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", default="fine_tune", type=str, help="the name of the saved file"
    )
    return parser.parse_args()


def train_an_epoch(model, train_dataloader, optimizer, loss, device):
    model.train()
    t_loss = 0
    for step, batch in enumerate(train_dataloader):
        prot_inputs, disease_inputs, label_inputs = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
        )
        optimizer.zero_grad()
        x = model.predict(prot_inputs, disease_inputs)
        logits = model.dropout(x)
        output = loss(logits, label_inputs.view(-1).to(torch.int64))
        t_loss += output.item()
        output.backward()
        optimizer.step()
    return t_loss


def evaluate(model, test_dataloader, device):
    model.eval()
    predicted_labels, target_labels = list(), list()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            prot_inputs, disease_inputs, label_inputs = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            logits = model.predict(prot_inputs, disease_inputs)
            predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
            target_labels.extend(label_inputs.cpu().view(-1).detach().numpy())
        predicted_labels, target_labels = (
            np.array(predicted_labels),
            np.array(target_labels),
        )
        roc_auc_score = metrics.roc_auc_score(target_labels, predicted_labels)
        average_precision_score = metrics.average_precision_score(
            target_labels, predicted_labels
        )
    return roc_auc_score, average_precision_score


def train(args):
    # defining parameters
    if args.model_path:
        args.model_short = (
            args.model_path.split("/")[-3] + "_" + args.model_path.split("/")[-2]
        )
        print(f"model name {args.model_short}")
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )

    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)

    def collate_fn_batch_encoding(batch):
        query1, query2, scores = zip(*batch)
        query_encodings1 = prot_tokenizer.batch_encode_plus(
            list(query1),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        query_encodings2 = disease_tokenizer.batch_encode_plus(
            list(query2),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        scores = torch.tensor(list(scores))
        return query_encodings1["input_ids"], query_encodings2["input_ids"], scores

    prot_model = BertModel.from_pretrained(args.prot_encoder_path).to(args.device)
    disease_model = BertModel.from_pretrained(args.disease_encoder_path).to(args.device)
    model = GDA_Metric_Learning(prot_model, disease_model, 1024, 768, args)
    if args.use_adapter:
        # print(model)
        # model.init_adapters(reduction_factor=args.reduction_factor)
        if args.model_path:
            prot_model_path = os.path.join(
                args.model_path, f"prot_adapter_step_{args.step}"
            )
            disease_model_path = os.path.join(
                args.model_path, f"disease_adapter_step_{args.step}"
            )
            model.load_adapters(prot_model_path, disease_model_path)
            print(f"loaded prior model {args.model_path}.")
        else:
            model.init_adapters(reduction_factor=args.reduction_factor)

    # print(model)
    model.add_classification_head(out_dim=2)
    model = model.to(args.device)
    wandb.config.update(args)
    # model.freeze_encoders(args.freeze_prot_encoder, args.freeze_disease_encoder)
    disGeNET = DisGeNETProcessor()
    train_examples = disGeNET.get_train_examples(args.test)
    print(f"get training examples: {len(train_examples)}")
    # Validation data_loader
    valid_examples = disGeNET.get_dev_examples(args.test)
    print(f"get validation examples: {len(valid_examples)}")
    # Test data_loader
    test_examples = disGeNET.get_test_examples(args.test)
    print(f"get test examples: {len(test_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_batch_encoding,
    )
    valid_dataloader = DataLoader(
        valid_examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_batch_encoding,
    )
    test_dataloader = DataLoader(
        test_examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_batch_encoding,
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01
    )
    loss = F.cross_entropy
    best_dev_auc = 0
    unimproved_iters = 0
    for epoch in tqdm(range(args.max_epoch), desc="Training"):
        epoch_loss = train_an_epoch(
            model, train_dataloader, optimizer, loss, args.device
        )
        valid_scores = evaluate(model, valid_dataloader, args.device)
        test_scores = evaluate(model, test_dataloader, args.device)
        print("epoch_loss:\t", epoch_loss)
        print("valid_auc:\t", valid_scores[0])
        print("test_auc:\t", test_scores[0])
        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "valid_auc": valid_scores[0],
                "test_auc": test_scores[0],
                "valid_ap": valid_scores[1],
                "test_ap": test_scores[1],
            }
        )
        # Update validation results
        if valid_scores[0] > best_dev_auc:
            unimproved_iters = 0
            best_dev_auc = valid_scores[0]
            # torch.save(model, args.save_name + "model.bin")
            wandb.log(
                {
                    "best_valid_auc": valid_scores[0],
                    "best_test_auc": test_scores[0],
                    "best_valid_ap": valid_scores[1],
                    "best_test_ap": test_scores[1],
                }
            )
        else:
            unimproved_iters += 1
            if unimproved_iters >= args.patience:
                print(f"Early Stopped on Epoch: {epoch}, Best Dev MSE: {best_dev_auc}")
                break


if __name__ == "__main__":
    args = parse_config()
    if torch.cuda.is_available():
        print("cuda is available.")
        print(f"current device {args}.")
    else:
        args.device = "cpu"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
    best_model_dir = (
        f"{args.save_path_prefix}{args.save_name}_{timestamp_str}_{random_str}/"
    )
    os.makedirs(best_model_dir)
    args.save_name = best_model_dir
    train(args)
