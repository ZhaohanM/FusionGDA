import argparse
import os
import random
import string
import sys
from datetime import datetime

sys.path.append("../")
import torch
import torch.nn as nn
from src.utils.tdc_disgenet_processor import  (
    DisGeNETProcessor,
    convert_examples_to_tokens,
    convert_tokens_to_tensors,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.metric_learning_models import GDA_Metric_Learning

import wandb

wandb.init(project="july_finetune_gda_with_adapter")


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
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epoch", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="../../save_model_ckp/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", default="fine_tune", type=str, help="the name of the saved file"
    )
    return parser.parse_args()


def train_an_epoch(model, train_dataloader, optimizer, loss):
    model.train()
    t_loss = 0
    for step, batch in enumerate(train_dataloader):
        prot_input, disease_inputs, label_inputs = batch
        optimizer.zero_grad()
        out = model.predict(prot_input, disease_inputs)
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
            out = model.predict(prot_input, disease_inputs)
            output = metric(out, label_inputs)
            metric_val += output.item()
            count += label_inputs.size()[0]
    return round(metric_val / count, 4)


def train(args):
    # defining parameters
    args.model_short = (
        args.model_path.split("/")[-3] + "_" + args.model_path.split("/")[-2]
    )
    print(f"model name {args.model_short}")
    prot_tokenizer = BertTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )

    disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)

    prot_model = BertModel.from_pretrained(args.prot_encoder_path).to(args.device)
    disease_model = BertModel.from_pretrained(args.disease_encoder_path).to(args.device)
    model = GDA_Metric_Learning(prot_model, disease_model, 1024, 768, args)
    if args.model_path:
        # print(model)
        # model.init_adapters(reduction_factor=args.reduction_factor)
        prot_model_path = os.path.join(
            args.model_path, f"prot_adapter_step_{args.step}"
        )
        disease_model_path = os.path.join(
            args.model_path, f"disease_adapter_step_{args.step}"
        )
        model.load_adapters(prot_model_path, disease_model_path)
        print(f"loaded prior model {args.model_path}.")
    # print(model)
    model.add_regression_head()
    model = model.to(args.device)
    print(f"loaded prior model {args.model_path}.")
    wandb.config.update(args)
    model.freeze_encoders(args.freeze_prot_encoder, args.freeze_disease_encoder)
    model.train()
    disGeNET = DisGeNETProcessor()
    examples = disGeNET.get_train_examples(args.test)
    print(f"get training examples: {len(examples[0])}")

    # tokens = convert_examples_to_tokens(examples, prot_tokenizer, disease_tokenizer, test=True)  ## Trun the test off if doing the real training
    tokens = convert_examples_to_tokens(
        args, examples, prot_tokenizer, disease_tokenizer
    )

    inputs = convert_tokens_to_tensors(tokens, args.device)
    train_data = TensorDataset(
        inputs["prot_inputs"], inputs["disease_inputs"], inputs["label_inputs"]
    )
    train_sampler = RandomSampler(train_data)

    # Validation data_loader
    valid_examples = disGeNET.get_dev_examples(args.test)
    print(f"get validate examples: {len(valid_examples[0])}")
    # valid_tokens = convert_examples_to_tokens(valid_examples, prot_tokenizer, disease_tokenizer, test=True)  ## Trun the test off if doing the real training
    valid_tokens = convert_examples_to_tokens(
        args, valid_examples, prot_tokenizer, disease_tokenizer
    )
    valid_inputs = convert_tokens_to_tensors(valid_tokens, args.device)
    valid_data = TensorDataset(
        valid_inputs["prot_inputs"],
        valid_inputs["disease_inputs"],
        valid_inputs["label_inputs"],
    )
    valid_sampler = RandomSampler(valid_data)

    # Test data_loader
    test_examples = disGeNET.get_test_examples(args.test)
    print(f"get test examples: {len(test_examples[0])}")
    # test_tokens = convert_examples_to_tokens(test_examples, prot_tokenizer, disease_tokenizer, test=True)  ## Trun the test off if doing the real training
    test_tokens = convert_examples_to_tokens(
        args, test_examples, prot_tokenizer, disease_tokenizer
    )

    test_inputs = convert_tokens_to_tensors(test_tokens, args.device)
    test_data = TensorDataset(
        test_inputs["prot_inputs"],
        test_inputs["disease_inputs"],
        test_inputs["label_inputs"],
    )
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size
    )
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=args.batch_size
    )
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.batch_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss = nn.MSELoss()
    metric = nn.MSELoss(reduction="sum")
    best_dev_mse = sys.maxsize
    unimproved_iters = 0
    for epoch in tqdm(range(args.max_epoch), desc="Training"):
        epoch_loss = train_an_epoch(model, train_dataloader, optimizer, loss)
        print("epoch_loss:\t", epoch_loss)
        valid_mse = evaluate(model, valid_dataloader, optimizer, metric)
        test_mse = evaluate(model, test_dataloader, optimizer, metric)
        print("valid_mse:\t", valid_mse)
        print("test_mse:\t", test_mse)
        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "valid_mse": valid_mse,
                "test_mse": test_mse,
            }
        )
        # Update validation results
        if valid_mse < best_dev_mse:
            unimproved_iters = 0
            best_dev_mse = valid_mse
            torch.save(model, args.save_name + "model.bin")
            wandb.log(
                {
                    "best_valid_mse": valid_mse,
                    "best_test_mse": test_mse,
                }
            )
        else:
            unimproved_iters += 1
            if unimproved_iters >= args.patience:
                early_stop = True
                print(f"Early Stopped on Epoch: {epoch}, Best Dev MSE: {best_dev_mse}")
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
