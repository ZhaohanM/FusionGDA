import logging
import os
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from torch.cuda.amp import autocast
from torch.nn import Module
from tqdm import tqdm
from transformers import AdapterConfig
from transformers.adapters.composition import Fuse, Stack

from src.utils.gd_model import GDANet

LOGGER = logging.getLogger(__name__)


class GDA_Metric_Learning(GDANet):
    def __init__(
        self, prot_encoder, disease_encoder, prot_out_dim, disease_out_dim, args
    ):
        """Constructor for the model.

        Args:
            prot_encoder (_type_): Protein encoder.
            disease_encoder (_type_): Disease Textual encoder.
            prot_out_dim (_type_): Dimension of the Protein encoder.
            disease_out_dim (_type_): Dimension of the Disease encoder.
            args (_type_): _description_
        """
        super(GDA_Metric_Learning, self).__init__(
            prot_encoder,
            disease_encoder,
        )
        self.prot_encoder = prot_encoder
        self.disease_encoder = disease_encoder
        self.loss = args.loss
        self.use_miner = args.use_miner
        self.miner_margin = args.miner_margin
        self.agg_mode = args.agg_mode
        self.prot_reg = nn.Linear(prot_out_dim, disease_out_dim)
        self.prot_adapter_name = None
        self.disease_adapter_name = None
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(
                margin=args.miner_margin, type_of_triplets="all"
            )
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(
                alpha=1, beta=60, base=0.5
            )  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(
                temperature=0.07
            )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        self.fusion = False
        self.stack = False
        self.dropout = torch.nn.Dropout(args.dropout)
        print("miner:", self.miner)
        print("loss:", self.loss)

    def add_fusion(self):
        adapter_setup = Fuse("prot_adapter", "disease_adapter")
        self.prot_encoder.add_fusion(adapter_setup)
        self.prot_encoder.set_active_adapters(adapter_setup)
        self.prot_encoder.train_fusion(adapter_setup)
        self.disease_encoder.add_fusion(adapter_setup)
        self.disease_encoder.set_active_adapters(adapter_setup)
        self.disease_encoder.train_fusion(adapter_setup)
        self.fusion = True

    def add_stack_gda(self, reduction_factor):
        self.add_gda_adapters(reduction_factor=reduction_factor)
        # adapter_setup = Fuse("prot_adapter", "disease_adapter")
        self.prot_encoder.active_adapters = Stack(
            self.prot_adapter_name, self.gda_adapter_name
        )
        self.disease_encoder.active_adapters = Stack(
            self.disease_adapter_name, self.gda_adapter_name
        )
        print("stacked adapters loaded.")
        self.stack = True

    def load_adapters(
        self,
        prot_model_path,
        disease_model_path,
        prot_adapter_name="prot_adapter",
        disease_adapter_name="disease_adapter",
    ):
        if os.path.exists(prot_model_path):
            print(f"loading prot adapter from: {prot_model_path}")
            self.prot_adapter_name = prot_adapter_name
            self.prot_encoder.load_adapter(prot_model_path, load_as=prot_adapter_name)
            self.prot_encoder.set_active_adapters(prot_adapter_name)
            print(f"load protein adapters from: {prot_model_path} {prot_adapter_name}")
        else:
            print(f"{prot_model_path} not exits")

        if os.path.exists(disease_model_path):
            print(f"loading prot adapter from: {disease_model_path}")
            self.disease_adapter_name = disease_adapter_name
            self.disease_encoder.load_adapter(
                disease_model_path, load_as=disease_adapter_name
            )
            self.disease_encoder.set_active_adapters(disease_adapter_name)
            print(
                f"load disease adapters from: {disease_model_path} {disease_adapter_name}"
            )
        else:
            print(f"{disease_model_path} not exits")

    def add_gda_adapters(
        self,
        gda_adapter_name="gda_adapter",
        reduction_factor=16,
    ):
        """Initialise adapters

        Args:
            prot_adapter_name (str, optional): _description_. Defaults to "prot_adapter".
            disease_adapter_name (str, optional): _description_. Defaults to "disease_adapter".
            reduction_factor (int, optional): _description_. Defaults to 16.
        """
        adapter_config = AdapterConfig.load(
            "pfeiffer", reduction_factor=reduction_factor
        )
        self.gda_adapter_name = gda_adapter_name
        self.prot_encoder.add_adapter(gda_adapter_name, config=adapter_config)
        self.prot_encoder.train_adapter([gda_adapter_name])
        self.disease_encoder.add_adapter(gda_adapter_name, config=adapter_config)
        self.disease_encoder.train_adapter([gda_adapter_name])

    def init_adapters(
        self,
        prot_adapter_name="gda_prot_adapter",
        disease_adapter_name="gda_disease_adapter",
        reduction_factor=16,
    ):
        """Initialise adapters

        Args:
            prot_adapter_name (str, optional): _description_. Defaults to "prot_adapter".
            disease_adapter_name (str, optional): _description_. Defaults to "disease_adapter".
            reduction_factor (int, optional): _description_. Defaults to 16.
        """
        adapter_config = AdapterConfig.load(
            "pfeiffer", reduction_factor=reduction_factor
        )
        self.prot_adapter_name = prot_adapter_name
        self.disease_adapter_name = disease_adapter_name
        self.prot_encoder.add_adapter(prot_adapter_name, config=adapter_config)
        self.prot_encoder.train_adapter([prot_adapter_name])
        self.disease_encoder.add_adapter(disease_adapter_name, config=adapter_config)
        self.disease_encoder.train_adapter([disease_adapter_name])
        print(f"adapter modules initialized")

    def save_adapters(self, save_path_prefix, total_step):
        """Save adapters into file.

        Args:
            save_path_prefix (string): saving path prefix.
            total_step (int): total step number.
        """
        prot_save_dir = os.path.join(
            save_path_prefix, f"prot_adapter_step_{total_step}"
        )
        disease_save_dir = os.path.join(
            save_path_prefix, f"disease_adapter_step_{total_step}"
        )
        os.makedirs(prot_save_dir, exist_ok=True)
        os.makedirs(disease_save_dir, exist_ok=True)
        self.prot_encoder.save_adapter(prot_save_dir, self.prot_adapter_name)
        prot_head_save_path = os.path.join(prot_save_dir, "prot_head.bin")
        torch.save(self.prot_reg, prot_head_save_path)
        self.disease_encoder.save_adapter(disease_save_dir, self.disease_adapter_name)
        if self.fusion:
            self.prot_encoder.save_all_adapters(prot_save_dir)
            self.disease_encoder.save_all_adapters(disease_save_dir)

    def predict(self, x1, x2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """

        x1 = self.prot_encoder(x1)[0][:, 0]
        x2 = self.disease_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        if self.reg is not None:
            x = self.reg(x)
            return x
        if self.cls is not None:
            x = self.cls(x)
            return x

    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """

        last_hidden_state1 = self.prot_encoder(
            query_toks1, return_dict=True
        ).last_hidden_state
        last_hidden_state1 = self.prot_reg(
            last_hidden_state1
        )  # transform the prot embedding into the same dimension as the disease embedding
        last_hidden_state2 = self.disease_encoder(
            query_toks2, return_dict=True
        ).last_hidden_state
        if self.agg_mode == "cls":
            query_embed1 = last_hidden_state1[:, 0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (
                last_hidden_state1 * query_toks1["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks1["attention_mask"].sum(-1).unsqueeze(-1)
            query_embed2 = (
                last_hidden_state2 * query_toks2["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks2["attention_mask"].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        # Generate labels for positive samples from 0 to N in order for the loss function to generate negative samples.
        labels = torch.cat([torch.arange(len(labels)), torch.arange(len(labels))], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


class DDA_Metric_Learning(Module):
    def __init__(self, disease_encoder, args):
        """Constructor for the model.

        Args:
            disease_encoder (_type_): disease encoder.
            args (_type_): _description_
        """
        super(DDA_Metric_Learning, self).__init__()
        self.disease_encoder = disease_encoder
        self.loss = args.loss
        self.use_miner = args.use_miner
        self.miner_margin = args.miner_margin
        self.agg_mode = args.agg_mode
        self.disease_adapter_name = None
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(
                margin=args.miner_margin, type_of_triplets="all"
            )
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(
                alpha=1, beta=60, base=0.5
            )  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(
                temperature=0.07
            )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        self.reg = None
        self.cls = None
        self.dropout = torch.nn.Dropout(args.dropout)
        print("miner:", self.miner)
        print("loss:", self.loss)

    def add_classification_head(self, disease_out_dim=768, out_dim=2):
        """Add regression head.

        Args:
            disease_out_dim (_type_): disease encoder output dimension.
            out_dim (int, optional): output dimension. Defaults to 2.
            drop_out (int, optional): dropout rate. Defaults to 0.
        """
        self.cls = nn.Linear(disease_out_dim * 2, out_dim)

    def load_disease_adapter(
        self,
        disease_model_path,
        disease_adapter_name="disease_adapter",
    ):
        if os.path.exists(disease_model_path):
            self.disease_adapter_name = disease_adapter_name
            self.disease_encoder.load_adapter(
                disease_model_path, load_as=disease_adapter_name
            )
            self.disease_encoder.set_active_adapters(disease_adapter_name)
            print(
                f"load disease adapters from: {disease_model_path} {disease_adapter_name}"
            )
        else:
            print(f"{disease_adapter_name} not exits")

    def init_adapters(
        self,
        disease_adapter_name="disease_adapter",
        reduction_factor=16,
    ):
        """Initialise adapters

        Args:
            disease_adapter_name (str, optional): _description_. Defaults to "disease_adapter".
            reduction_factor (int, optional): _description_. Defaults to 16.
        """
        adapter_config = AdapterConfig.load(
            "pfeiffer", reduction_factor=reduction_factor
        )
        self.disease_adapter_name = disease_adapter_name
        self.disease_encoder.add_adapter(disease_adapter_name, config=adapter_config)
        self.disease_encoder.train_adapter([disease_adapter_name])

    def save_adapters(self, save_path_prefix, total_step):
        """Save adapters into file.

        Args:
            save_path_prefix (string): saving path prefix.
            total_step (int): total step number.
        """
        disease_save_dir = os.path.join(
            save_path_prefix, f"disease_adapter_step_{total_step}"
        )
        os.makedirs(disease_save_dir, exist_ok=True)
        self.disease_encoder.save_adapter(disease_save_dir, self.disease_adapter_name)

    def predict(self, x1, x2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        x1 = self.disease_encoder(x1)[0][:, 0]
        x2 = self.disease_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        return x

    def module_predict(self, x1, x2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        x1 = self.disease_encoder.module(x1)[0][:, 0]
        x2 = self.disease_encoder.module(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        return x

    @autocast()
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        last_hidden_state1 = self.disease_encoder(
            **query_toks1, return_dict=True
        ).last_hidden_state
        last_hidden_state2 = self.disease_encoder(
            **query_toks2, return_dict=True
        ).last_hidden_state
        if self.agg_mode == "cls":
            query_embed1 = last_hidden_state1[:, 0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (
                last_hidden_state1 * query_toks1["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks1["attention_mask"].sum(-1).unsqueeze(-1)
            query_embed2 = (
                last_hidden_state2 * query_toks2["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks2["attention_mask"].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        # Generate labels for positive samples from 0 to N in order for the loss function to generate negative samples.
        labels = torch.cat([torch.arange(len(labels)), torch.arange(len(labels))], dim=0)
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)


class PPI_Metric_Learning(Module):
    def __init__(self, prot_encoder, args):
        """Constructor for the model.

        Args:
            prot_encoder (_type_): Protein encoder.
            prot_encoder (_type_): prot Textual encoder.
            prot_out_dim (_type_): Dimension of the Protein encoder.
            prot_out_dim (_type_): Dimension of the prot encoder.
            args (_type_): _description_
        """
        super(PPI_Metric_Learning, self).__init__()
        self.prot_encoder = prot_encoder
        self.loss = args.loss
        self.use_miner = args.use_miner
        self.miner_margin = args.miner_margin
        self.agg_mode = args.agg_mode
        self.prot_adapter_name = None
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(
                margin=args.miner_margin, type_of_triplets="all"
            )
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(
                alpha=1, beta=60, base=0.5
            )  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(
                temperature=0.07
            )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        self.reg = None
        self.cls = None
        self.dropout = torch.nn.Dropout(args.dropout)
        print("miner:", self.miner)
        print("loss:", self.loss)

    def add_classification_head(self, prot_out_dim=1024, out_dim=2):
        """Add regression head.

        Args:
            prot_out_dim (_type_): protein encoder output dimension.
            disease_out_dim (_type_): disease encoder output dimension.
            out_dim (int, optional): output dimension. Defaults to 2.
            drop_out (int, optional): dropout rate. Defaults to 0.
        """
        self.cls = nn.Linear(prot_out_dim + prot_out_dim, out_dim)

    def load_prot_adapter(
        self,
        prot_model_path,
        prot_adapter_name="prot_adapter",
    ):
        if os.path.exists(prot_model_path):
            self.prot_adapter_name = prot_adapter_name
            self.prot_encoder.load_adapter(prot_model_path, load_as=prot_adapter_name)
            self.prot_encoder.set_active_adapters(prot_adapter_name)
            print(f"load protein adapters from: {prot_model_path} {prot_adapter_name}")
        else:
            print(f"{prot_model_path} not exits")

    def init_adapters(
        self,
        prot_adapter_name="prot_adapter",
        reduction_factor=16,
    ):
        """Initialise adapters

        Args:
            prot_adapter_name (str, optional): _description_. Defaults to "prot_adapter".
            reduction_factor (int, optional): _description_. Defaults to 16.
        """
        adapter_config = AdapterConfig.load(
            "pfeiffer", reduction_factor=reduction_factor
        )
        self.prot_adapter_name = prot_adapter_name
        self.prot_encoder.add_adapter(prot_adapter_name, config=adapter_config)
        self.prot_encoder.train_adapter([prot_adapter_name])

    def save_adapters(self, save_path_prefix, total_step):
        """Save adapters into file.

        Args:
            save_path_prefix (string): saving path prefix.
            total_step (int): total step number.
        """
        prot_save_dir = os.path.join(
            save_path_prefix, f"prot_adapter_step_{total_step}"
        )
        os.makedirs(prot_save_dir, exist_ok=True)
        self.prot_encoder.save_adapter(prot_save_dir, self.prot_adapter_name)

    def predict(self, x1, x2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        x1 = self.prot_encoder(x1)[0][:, 0]
        x2 = self.prot_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        return x

    def module_predict(self, x1, x2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        x1 = self.prot_encoder.module(x1)[0][:, 0]
        x2 = self.prot_encoder.module(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        return x

    @autocast()
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        last_hidden_state1 = self.prot_encoder(
            **query_toks1, return_dict=True
        ).last_hidden_state
        last_hidden_state2 = self.prot_encoder(
            **query_toks2, return_dict=True
        ).last_hidden_state
        if self.agg_mode == "cls":
            query_embed1 = last_hidden_state1[:, 0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (
                last_hidden_state1 * query_toks1["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks1["attention_mask"].sum(-1).unsqueeze(-1)
            query_embed2 = (
                last_hidden_state2 * query_toks2["attention_mask"].unsqueeze(-1)
            ).sum(1) / query_toks2["attention_mask"].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        # Generate labels for positive samples from 0 to N in order for the loss function to generate negative samples.
        labels = torch.cat([torch.arange(len(labels)), torch.arange(len(labels))], dim=0)
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)
