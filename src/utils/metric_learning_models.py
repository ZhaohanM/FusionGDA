import logging
import os
import sys

sys.path.append("../")
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses


import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_metric_learning import losses, miners
from torch.cuda.amp import autocast
from torch.nn import Module
from tqdm import tqdm
from transformers import AdapterConfig
from transformers.adapters.composition import Fuse, Stack
from utils.gd_model import GDANet
from torch.nn import MultiheadAttention

LOGGER = logging.getLogger(__name__)


class FusionModule(nn.Module):
    def __init__(self, out_dim, num_head, dropout= 0.1):
        super(FusionModule, self).__init__()
        """FusionModule.

        Args:
            dropout= 0.1 is defaut
            out_dim: model output dimension
            num_head = 8: Multi-head Attention
        """

        self.out_dim = out_dim
        self.num_head = num_head

        self.WqS = nn.Linear(out_dim, out_dim)
        self.WkS = nn.Linear(out_dim, out_dim)
        self.WvS = nn.Linear(out_dim, out_dim)

        self.WqT = nn.Linear(out_dim, out_dim)
        self.WkT = nn.Linear(out_dim, out_dim)
        self.WvT = nn.Linear(out_dim, out_dim)
        self.multi_head_attention = nn.MultiheadAttention(out_dim, num_head, dropout=dropout)

    def forward(self, protein_representation, disease_representation):
        #  nn.MultiheadAttention The input representation is (token_length, batch_size, out_dim)
        zs = protein_representation.permute(1, 0, 2)
        zt = disease_representation.permute(1, 0, 2)   
        
        # Compute query, key and value representations
        qs = self.WqS(zs)
        ks = self.WkS(zs)
        vs = self.WvS(zs)

        qt = self.WqT(zt)
        kt = self.WkT(zt)
        vt = self.WvT(zt)
        
        #self.multi_head_attention() The function returns two values: the representation and the attention weight matrix, computed after multiple attentions. In this case, we only care about the computed representation and not the attention weight matrix, so "_" is used to indicate that we do not intend to use or store the second return value.
        zs_attention1, _ = self.multi_head_attention(qs, ks, vs)
        zs_attention2, _ = self.multi_head_attention(qs, kt, vt)
        zt_attention1, _ = self.multi_head_attention(qt, kt, vt)
        zt_attention2, _ = self.multi_head_attention(qt, ks, vs)
        


        protein_fused = 0.5 * (zs_attention1 + zs_attention2)
        dis_fused = 0.5 * (zt_attention1 + zt_attention2)
        
        #print("protein_fused",protein_fused.shape,"dis_fused",dis_fused.shape)
        return protein_fused, dis_fused

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
        
        self.fusion_layer = FusionModule(disease_out_dim, num_head=8)
        # MMP Prediction Heads
        self.prot_pred_head = nn.Sequential(
            nn.Linear(disease_out_dim, disease_out_dim),
            nn.ReLU(),
            nn.Linear(disease_out_dim, 30)  #vocabulary size : prot model tokenize length
        )
        self.dise_pred_head = nn.Sequential(
            nn.Linear(disease_out_dim, disease_out_dim),
            nn.ReLU(),
            nn.Linear(disease_out_dim, 30522) #vocabulary size : disease model tokenize length
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(
                margin=args.miner_margin, type_of_triplets="all"
            )
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(
                alpha=2, beta=50, base=0.5
            )  # 1,2,3; 40,50,60
            #1_40=1.5141 50=1.4988 60=1.4905 2_60=1.1786 50=1.1874 40=1.2008 3_40=1.1146 50=1.1012
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss(
                m=0.4, gamma=80
            )
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss( 
                margin=0.05, swap=False, smooth_loss=False,
                triplets_per_anchor="all")
                
           #losses.TripletMarginLoss(distance = CosineSimilarity(), 
                                    #reducer = ThresholdReducer(high=0.3), 
                                    #embedding_regularizer = LpRegularizer())
                
               # margin=0.05, swap=False, smooth_loss=False,
               # triplets_per_anchor="all"
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(
                temperature=0.07
            )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss(
                neg_margin=1, pos_margin=0
            )
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss(
                softmax_scale=1
            )
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
            
    def non_adapters(
            self,
            prot_model_path,
            disease_model_path,
            
    ):
        if os.path.exists(prot_model_path):
            # Load the entire model for prot_model
            prot_model = torch.load(prot_model_path)
            # Set the prot_encoder to the loaded model
            self.prot_encoder = prot_model.prot_encoder
            print(f"load protein from: {prot_model_path}")     
        else:
            print(f"{prot_model_path} not exits")

        if os.path.exists(disease_model_path):
            # Load the entire model for disease_model
            disease_model = torch.load(disease_model_path)
            # Set the disease_encoder to the loaded model
            self.disease_encoder = disease_model.disease_encoder
            print(f"load disease from: {disease_model_path}")

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

    def predict(self, query_toks1, query_toks2):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        # print("query_toks1 =", query_toks1.shape, "query_toks2 =", query_toks2.shape)
        last_hidden_state1 = self.prot_encoder(
            query_toks1, return_dict=True
        ).last_hidden_state
        print("last_hidden_state1 =", last_hidden_state1.shape)
        last_hidden_state2 = self.disease_encoder(
            query_toks2, return_dict=True
        ).last_hidden_state
        #print("last_hidden_state2 =", last_hidden_state2.shape)
        
        #print("last_hidden_state1 =", last_hidden_state1.shape)
       # Apply the fusion layer and Recovery of representational shape
        prot_fused1, dis_fused1 = self.fusion_layer(last_hidden_state1, last_hidden_state2)
        #print("prot_fused1 :", prot_fused1.shape) 
        prot_fused = prot_fused1.permute(1, 0, 2)
        dis_fused = dis_fused1.permute(1, 0, 2)
        #print("prot_fused :", prot_fused.shape)
        
        # Multi-modal Mask Prediction (MMP)
        prot_pred = self.prot_pred_head(prot_fused) # [12, 512, 768]
        dise_pred = self.dise_pred_head(dis_fused) # [12, 512, 768]
        
        if self.agg_mode == "cls":
            query_embed1 = prot_pred[:, 0]  # query : [batch_size, hidden]
            query_embed2 = dise_pred[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = prot_pred.mean(1)  # query : [batch_size, hidden]
            query_embed2 = dise_pred.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (
                                   prot_fused * query_toks1["attention_mask"].unsqueeze(-1)
                           ).sum(1) / query_toks1["attention_mask"].sum(-1).unsqueeze(-1)
            query_embed2 = (
                                   dis_fused * query_toks2["attention_mask"].unsqueeze(-1)
                           ).sum(1) / query_toks2["attention_mask"].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        #print("query_embed1 =", query_embed1.shape, "query_embed2 =", query_embed2.shape)
        query_embed = torch.cat([query_embed1, query_embed2], dim=1)
        #print("query_embed =", query_embed.shape)
  
    
        # MLP
        if self.reg is not None:
            x = self.reg(query_embed)
            print("x_reg = ", len(x), x)
            return x
        if self.cls is not None:
            x = self.cls(query_embed)
            print("x_cls = ", len(x), x)
            return x

    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        """
        #print("query_toks1 =", query_toks1.shape, "query_toks2 =", query_toks2.shape)
        #print("lables", len(lables), lables)
        last_hidden_state1 = self.prot_encoder(
            query_toks1, return_dict=True
        ).last_hidden_state
        # print("last_hidden_state1 =", last_hidden_state1.shape)
        last_hidden_state1 = self.prot_reg(
            last_hidden_state1
        )  # transform the prot embedding into the same dimension as the disease embedding
        last_hidden_state2 = self.disease_encoder(
            query_toks2, return_dict=True
        ).last_hidden_state
        
        
       #print("last_hidden_state1 =", last_hidden_state1.shape)
       # Apply the fusion layer and Recovery of representational shape
        prot_fused1, dis_fused1 = self.fusion_layer(last_hidden_state1, last_hidden_state2)
        #print("prot_fused1 :", prot_fused1.shape) 
        prot_fused = prot_fused1.permute(1, 0, 2)
        dis_fused = dis_fused1.permute(1, 0, 2)
        #print("prot_fused :", prot_fused.shape)

       # Multi-modal Mask Prediction (MMP)
        prot_pred = self.prot_pred_head(prot_fused) # [12, 512, 768]
        dise_pred = self.dise_pred_head(dis_fused) # [12, 512, 768]
        #print("prot_pred:", prot_pred.shape)
        #print("dise_pred:", dise_pred.shape)

        if self.agg_mode == "cls":
            query_embed1 = prot_pred[:, 0]  # query : [batch_size, hidden]
            query_embed2 = dise_pred[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = prot_pred.mean(1)  # query : [batch_size, hidden]
            query_embed2 = dise_pred.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (
                                   prot_pred * query_toks1["attention_mask"].unsqueeze(-1)
                           ).sum(1) / query_toks1["attention_mask"].sum(-1).unsqueeze(-1)
            query_embed2 = (
                                   dis_fused * query_toks2["attention_mask"].unsqueeze(-1)
                           ).sum(1) / query_toks2["attention_mask"].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        #print("query_embed1 =", query_embed1.shape, "query_embed2 =", query_embed2.shape)
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        #print("query_embed =", len(query_embed))
        labels = torch.cat([torch.arange(len(labels)), torch.arange(len(labels))], dim=0)
        #print("lable =", len(labels), labels)
        
        # print("query_embed1 =", len(query_embed1), "query_embed2 =", len(query_embed2))
        # query_embed = torch.cat([query_embed1, query_embed2], dim=0)  
        
        # Compute Losses
        # prot_pred has the shape [12 * 512, vocab_size] and actual_tokens has the shape [12* 512] 
        loss_mask_prot = F.cross_entropy(prot_pred.view(-1, prot_pred.size(-1)), query_toks1.view(-1)) 
        # dise_pred has the shape [12 * 512, vocab_size] and actual_tokens has the shape [12* 512]
        loss_mask_dise = F.cross_entropy(dise_pred.view(-1, dise_pred.size(-1)), query_toks2.view(-1))

        loss_mmp = loss_mask_prot + loss_mask_dise
        
        #print("query_toks1:", query_toks1)
        #print("query_toks2:", query_toks2)
        
        print("loss_mask_prot:", loss_mask_prot)
        print("loss_mask_dise:", loss_mask_dise)
        print("loss_mmp:", loss_mmp)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) + loss_mmp
        else:
            loss = self.loss(query_embed, labels) + loss_mmp
            print('loss :', loss)
            return loss

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
