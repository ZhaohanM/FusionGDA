import argparse
import os
import random
import string
import sys
from datetime import datetime

sys.path.append("../")
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from src.utils.tdc_disgenet_processor import (
    DisGeNETProcessor,
    convert_examples_to_tokens,
    convert_tokens_to_tensors,
)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from src.utils.metric_learning_models import GDA_Metric_Learning

import wandb

wandb.init(project="july_finetune_gda_adapter_lightgbm")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path/name of the whole model",
    )
    parser.add_argument("--step", type=int, default=20)
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
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument(
        "--input_feature_save_path",
        type=str,
        default="../data/processed/",
        help="path of tokenized training data",
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_leaves", type=int, default=5)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
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


def get_feature(prot_model, disease_model, dataloader):
    """convert tensors of dataloader to embedding feature encoded by berts

    Args:
        prot_model (BertModel): Protein BERT model
        disease_model (BertModel): Textual BERT model
        dataloader (DataLoader): Dataloader

    Returns:
        (ndarray,ndarray): x, y
    """
    x = None
    y = None
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            prot_input, disease_inputs, y1 = batch
            # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
            prot_out = prot_model(prot_input)[0][:, 0, :].cpu()
            disease_out = disease_model(disease_inputs)[0][:, 0, :].cpu()
            x1 = np.concatenate((prot_out, disease_out), axis=1)
            if x is None:
                x = x1
            else:
                x = np.append(x, x1, axis=0)

            if y is None:
                y = y1.cpu().numpy()
            else:
                y = np.append(y, y1.cpu().numpy(), axis=0)
    return x, y


def train(args):
    # defining parameters
    args.model_short = (
        args.model_path.split("/")[-3] + "_" + args.model_path.split("/")[-2]
    )
    print(f"model name {args.model_short}")
    input_feat_file = os.path.join(
        args.input_feature_save_path,
        f"{args.model_short}_{args.step}_{args.test}_feat.npz",
    )
    if os.path.exists(input_feat_file):
        print(f"load prior feature data from {input_feat_file}.")
        loaded = np.load(input_feat_file)
        x_train, y_train = loaded["x_train"], loaded["y_train"]
        x_valid, y_valid = loaded["x_valid"], loaded["y_valid"]
        x_test, y_test = loaded["x_test"], loaded["y_test"]
    else:
        prot_tokenizer = BertTokenizer.from_pretrained(
            args.prot_encoder_path, do_lower_case=False
        )

        disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)

        prot_model = BertModel.from_pretrained(args.prot_encoder_path).to(args.device)
        disease_model = BertModel.from_pretrained(args.disease_encoder_path).to(
            args.device
        )
        if args.model_path:
            model = GDA_Metric_Learning(prot_model, disease_model, 1024, 768, args)
            # print(model)
            # model.init_adapters(reduction_factor=args.reduction_factor)
            prot_model_path = os.path.join(
                args.model_path, f"prot_adapter_step_{args.step}"
            )
            disease_model_path = os.path.join(
                args.model_path, f"disease_adapter_step_{args.step}"
            )
            model.load_adapters(prot_model_path, disease_model_path)
            model = model.to(args.device)
            prot_model = model.prot_encoder
            disease_model = model.disease_encoder
            print(f"loaded prior model {args.model_path}.")
        # print(model)
        disGeNET = DisGeNETProcessor()
        train_examples = disGeNET.get_train_examples(args.test)
        valid_examples = disGeNET.get_dev_examples(args.test)
        test_examples = disGeNET.get_test_examples(args.test)
        print(
            f"dataset loaded: train-{len(train_examples[0])}; valid-{len(valid_examples[0])}; test-{len(test_examples[0])}"
        )
        # tokens = convert_examples_to_tokens(examples, prot_tokenizer, disease_tokenizer, test=True)  ## Trun the test off if doing the real training
        train_tokens = convert_examples_to_tokens(
            args, train_examples, prot_tokenizer, disease_tokenizer
        )
        train_inputs = convert_tokens_to_tensors(train_tokens, args.device)
        train_data = TensorDataset(
            train_inputs["prot_inputs"],
            train_inputs["disease_inputs"],
            train_inputs["label_inputs"],
        )

        # valid dataloader
        valid_tokens = convert_examples_to_tokens(
            args, valid_examples, prot_tokenizer, disease_tokenizer
        )
        valid_inputs = convert_tokens_to_tensors(valid_tokens, args.device)
        valid_data = TensorDataset(
            valid_inputs["prot_inputs"],
            valid_inputs["disease_inputs"],
            valid_inputs["label_inputs"],
        )

        # Test data_loader
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

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.batch_size
        )
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=args.batch_size
        )
        test_sampler = SequentialSampler(test_data)

        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.batch_size
        )

        x_train, y_train = get_feature(prot_model, disease_model, train_dataloader)
        x_valid, y_valid = get_feature(prot_model, disease_model, valid_dataloader)
        x_test, y_test = get_feature(prot_model, disease_model, test_dataloader)
        print(x_train.shape)

        # Save input feature to reduce encoding time
        np.savez_compressed(
            input_feat_file,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            y_test=y_test,
        )
        print(f"save input feature into {input_feat_file}")
    wandb.config.update(args)
    print("train: ", x_train.shape, y_train.shape)
    print("valid: ", x_valid.shape, y_valid.shape)
    print("test: ", x_test.shape, y_test.shape)
    params = {
        "task": "train",
        "boosting": "gbdt",
        "objective": "regression",
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "metric": "l2",
        "verbose": 1,
    }
    wandb.config.update(params)
    # loading data
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    # fitting the model
    model = lgb.train(
        params, train_set=lgb_train, valid_sets=lgb_valid, early_stopping_rounds=30
    )
    # prediction
    y_pred = model.predict(x_test)

    # accuracy check
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (0.5)
    wandb.config.update({"rmse": rmse, "mse": mse})
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)


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
