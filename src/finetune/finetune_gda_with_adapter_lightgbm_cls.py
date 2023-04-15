import argparse
import os
import random
import string
import sys
from datetime import datetime

sys.path.append("../")
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, precision_recall_fscore_support
import numpy as np
import sklearn.metrics as metrics
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, AutoTokenizer, EsmModel
from utils.tdc_disgenet_processor import DisGeNETProcessor
from utils.metric_learning_models import GDA_Metric_Learning

wandb.init(project="Mar_Finetune_infoNCE_Fusion_MMP_TDC")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="path of the pretrained disease model located",
    )
    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="Rostlab/prot_bert_bfd",     
        #"facebook/galactica-6.7b", "Rostlab/prot_bert" "facebook/esm1b_t33_650M_UR50S" “facebook/esm2_t33_650M_UR50D”  facebook/galactica-6.7b
        help="path/name of protein encoder model located",
    )
    parser.add_argument(
        "--disease_encoder_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="path/name of textual pre-trained language model",
    )
    parser.add_argument("--reduction_factor", type=int, default=8)
    parser.add_argument(
        "--loss",
        help="{ms_loss|infoNCE|cosine_loss|circle_loss|triplet_loss}}",
        default="infoNCE",
    )
    parser.add_argument(
        "--input_feature_save_path",
        type=str,
        default="../../data/processed_infoNCE_fusion_mmp_TDC/",
        help="path of tokenized training data",
    )
    parser.add_argument(
        "--agg_mode", default="mean_all_tok", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_leaves", type=int, default=5)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--freeze_prot_encoder", action="store_true")
    parser.add_argument("--freeze_disease_encoder", action="store_true")
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--use_pooled", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--use_both_feature",
        help="use the both features of gnn_feature_v1_samples and pretrained models",
        action="store_true",
    )
    parser.add_argument(
        "--use_v1_feature_only",
        help="use the features of gnn_feature_v1_samples only",
        action="store_true",
    )
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


def get_feature(prot_model, disease_model, dataloader, args):
    """convert tensors of dataloader to embedding feature encoded by berts

    Args:
        prot_model (BertModel): Protein BERT model
        disease_model (BertModel): Textual BERT model
        dataloader (DataLoader): Dataloader

    Returns:
        (ndarray,ndarray): x, y
    """
    x = list()
    y = list()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            prot_input, disease_inputs, y1 = batch
            # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
            prot_input = prot_input.to(prot_model.device)
            disease_inputs = disease_inputs.to(disease_model.device)
            if args.use_pooled:
                prot_out = prot_model(prot_input).last_hidden_state.mean(1)
                disease_out = disease_model(disease_inputs,).last_hidden_state.mean(1)
            else:
                prot_out = prot_model(prot_input).last_hidden_state[:, 0]
                disease_out = disease_model(disease_inputs).last_hidden_state[:, 0]
            x1 = np.concatenate((prot_out.cpu(), disease_out.cpu()), axis=1)
            x.append(x1)
            y.append(y1.cpu().numpy())
    x = np.concatenate(x,axis=0)
    y = np.concatenate(y,axis=0)
    return x, y


def encode_pretrained_feature(args, disGeNET):
    input_feat_file = os.path.join(
        args.input_feature_save_path,
        f"{args.model_short}_{args.step}_{args.test}_use_{'pooled' if args.use_pooled else 'cls'}_feat.npz",
    )

    if os.path.exists(input_feat_file):
        print(f"load prior feature data from {input_feat_file}.")
        loaded = np.load(input_feat_file)
        x_train, y_train = loaded["x_train"], loaded["y_train"]
        x_valid, y_valid = loaded["x_valid"], loaded["y_valid"]
        x_test, y_test = loaded["x_test"], loaded["y_test"]
    else:
        #model, alphabet = esm.pretrained.load_model_and_alphabet(args.prot_encoder_path)
        #prot_tokenizer = AutoTokenizer.from_pretrained(args.prot_encoder_path)
        
        
        #model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        #batch_converter = alphabet.get_batch_converter()
        #prot_tokenizer = esm.Tokenizer(alphabet)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # prot_model = model.eval().to(device)

        prot_tokenizer = BertTokenizer.from_pretrained(args.prot_encoder_path, do_lower_case=False)
        disease_tokenizer = BertTokenizer.from_pretrained(args.disease_encoder_path)
        
        #prot_model = EsmModel.from_pretrained(args.prot_encoder_path)
        #prot_model = OPTForCausalLM.from_pretrained(args.prot_encoder_path, device_map="auto")
        prot_model = BertModel.from_pretrained(args.prot_encoder_path)
        disease_model = BertModel.from_pretrained(args.disease_encoder_path)
        
        if args.save_model_path:
            model = GDA_Metric_Learning(prot_model, disease_model, 768, 768, args)

            if args.use_adapter:
                prot_model_path = os.path.join(
                    args.save_model_path, f"prot_adapter_step_{args.step}"
                )
                disease_model_path = os.path.join(
                    args.save_model_path, f"disease_adapter_step_{args.step}"
                )
                model.load_adapters(prot_model_path, disease_model_path)
            else:
                prot_model_path = os.path.join(
                    args.save_model_path, f"step_{args.step}_model.bin"
                )
                disease_model_path = os.path.join(
                    args.save_model_path, f"step_{args.step}_model.bin"
                )
                model.non_adapters(prot_model_path, disease_model_path)
                
            model = model.to(args.device)
            prot_model = model.prot_encoder
            disease_model = model.disease_encoder
            print(f"loaded prior model {args.save_model_path}.")

                

        def collate_fn_batch_encoding(batch):
            #print(batch)
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
            collate_fn=collate_fn_batch_encoding,
        )
        valid_dataloader = DataLoader(
            valid_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )
        test_dataloader = DataLoader(
            test_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )
        print(
            f"dataset loaded: train-{len(train_examples)}; valid-{len(valid_examples)}; test-{len(test_examples)}"
        )
        # tokens = convert_examples_to_tokens(examples, prot_tokenizer, disease_tokenizer, test=True)  ## Trun the test off if doing the real training

        x_train, y_train = get_feature(prot_model, disease_model, train_dataloader, args)
        x_valid, y_valid = get_feature(prot_model, disease_model, valid_dataloader, args)
        x_test, y_test = get_feature(prot_model, disease_model, test_dataloader, args)
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
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def train(args):
    # defining parameters
    if args.save_model_path:
        args.model_short = (
            os.path.split("/")[-1] 
        )
        print(f"model name {args.model_short}")
    else:
        args.model_short = (
            args.disease_encoder_path.split("/")[-1] 
        )
        print(f"model name {args.model_short}")

    disGeNET = DisGeNETProcessor()
    wandb.config.update(args)
    # only use v1 feature
    if args.use_v1_feature_only:
        x_train, y_train = disGeNET.get_train_v1_feature()
        x_valid, y_valid = disGeNET.get_valid_v1_feature()
        x_test, y_test = disGeNET.get_test_v1_feature()
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = encode_pretrained_feature(
            args, disGeNET
        )

    if args.use_both_feature:
        x_train_v1_feature, y_train = disGeNET.get_train_v1_feature()
        x_valid_v1_feature, y_valid = disGeNET.get_valid_v1_feature()
        x_test_v1_feature, y_test = disGeNET.get_test_v1_feature()
        x_train = np.hstack((x_train, x_train_v1_feature))
        x_valid = np.hstack((x_valid, x_valid_v1_feature))
        x_test = np.hstack((x_test, x_test_v1_feature))

    print("train: ", x_train.shape, y_train.shape)
    print("valid: ", x_valid.shape, y_valid.shape)
    print("test: ", x_test.shape, y_test.shape)
    # Calculate the class weights
    #def apply_class_weights(y, class_weights):
    #    return class_weights.get(y)

    #class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    #class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    #vectorized_apply_class_weights = np.vectorize(apply_class_weights, excluded=['class_weights'])
    #sample_weights = vectorized_apply_class_weights(y_train, class_weights=class_weights)

    
    #positive_weight = len(y_train[y_train == 1])
    #negative_weight = len(y_train[y_train == 0])
    #total_samples = positive_weight + negative_weight
    #print("total_samples", total_samples)
    #scale_pos_weight = negative_weight / positive_weight
    #print("scale_pos_weight", scale_pos_weight)
    
    params = {
        "task": "train",
        "boosting": "gbdt",
        "objective": "binary",
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "metric": "binary_logloss", #"metric": "l2均方误差","binary_logloss交叉熵损失"  "auc"
        "verbose": 1,
    }
    wandb.config.update(params)
    #"class_weight": class_weight_dict, #the model will give more importance to the positive class
    
    
    # loading data      Number of positive: 3204, number of negative: 31930
    # 'scale_pos_weight': scale_pos_weight
    
    # Undersampling
    #rus = RandomUnderSampler(sampling_strategy={1: 3204, 0: 13204})   
    #x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
    
    # Oversampling
    #ros = RandomOverSampler(random_state=42)
    #x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

    # Oversampling with SMOTE 
    #smote = SMOTE(random_state=42)
    #x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
  
    lgb_train = lgb.Dataset(x_train, y_train) #(x_train_resampled, y_train_resampled) (x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    # fitting the model
    model = lgb.train(
        params, train_set=lgb_train, valid_sets=lgb_valid, early_stopping_rounds=30)
    
    # prediction
    valid_y_pred = model.predict(x_valid)
    test_y_pred = model.predict(x_test)
    
    # Accuracy
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # AUC
    valid_roc_auc_score = metrics.roc_auc_score(y_valid, valid_y_pred)
    valid_average_precision_score = metrics.average_precision_score(
        y_valid, valid_y_pred
    )
    test_roc_auc_score = metrics.roc_auc_score(y_test, test_y_pred)
    test_average_precision_score = metrics.average_precision_score(y_test, test_y_pred)
    
    # AUPR
    valid_aupr = metrics.average_precision_score(y_valid, valid_y_pred)
    test_aupr = metrics.average_precision_score(y_test, test_y_pred)

    # Fmax
    valid_precision, valid_recall, valid_thresholds = precision_recall_curve(y_valid, valid_y_pred)
    valid_fmax = (2 * valid_precision * valid_recall / (valid_precision + valid_recall)).max()
    test_precision, test_recall, test_thresholds = precision_recall_curve(y_test, test_y_pred)
    test_fmax = (2 * test_precision * test_recall / (test_precision + test_recall)).max()

    # F1
    valid_f1 = f1_score(y_valid, valid_y_pred >= 0.5)
    test_f1 = f1_score(y_test, test_y_pred >= 0.5)

    print(f"Validation AUPR: {valid_aupr:.4f}")
    print(f"Test AUPR: {test_aupr:.4f}")
    print(f"Validation Fmax: {valid_fmax:.4f}")
    print(f"Test Fmax: {test_fmax:.4f}")
    print(f"Validation F1: {valid_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    
    wandb.config.update(
        {
            "f1_score": test_f1,
            "fmax_score": test_fmax,
            "AUPR_score": test_aupr,
            "Accuracy": accuracy,
            "valid_AUC": valid_roc_auc_score,
            "valid_AP": valid_average_precision_score,
            "test_AUC": test_roc_auc_score,
            "test_AP": test_average_precision_score,
        }
    )
    print("valid roc_auc_score: %.4f" % valid_roc_auc_score)
    print("valid average_precision_score: %.4f" % valid_average_precision_score)
    print("test roc_auc_score: %.4f" % test_roc_auc_score)
    print("test average_precision_score: %.4f" % test_average_precision_score)


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
