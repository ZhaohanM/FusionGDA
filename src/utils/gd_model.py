import sys

import torch
import torch.nn as nn

sys.path.append("../")


class GDANet(torch.nn.Module):
    def __init__(
        self,
        prot_encoder,
        disease_encoder,
    ):
        """_summary_

        Args:
            prot_encoder (_type_): _description_
            disease_encoder (_type_): _description_
            prot_out_dim (int, optional): _description_. Defaults to 1024.
            disease_out_dim (int, optional): _description_. Defaults to 768.
            drop_out (int, optional): _description_. Defaults to 0.
            freeze_prot_encoder (bool, optional): _description_. Defaults to True.
            freeze_disease_encoder (bool, optional): _description_. Defaults to True.
        """
        super(GDANet, self).__init__()
        self.prot_encoder = prot_encoder
        self.disease_encoder = disease_encoder
        self.cls = None
        self.reg = None

    def add_regression_head(self, prot_out_dim=1024, disease_out_dim=768):
        """Add regression head.

        Args:
            prot_out_dim (_type_): protein encoder output dimension.
            disease_out_dim (_type_): disease encoder output dimension.
            drop_out (int, optional): dropout rate. Defaults to 0.
        """
        self.reg = nn.Linear(prot_out_dim + disease_out_dim, 1)

    def add_classification_head(
        self, prot_out_dim=1024, disease_out_dim=768, out_dim=2
    ):
        """Add regression head.

        Args:
            prot_out_dim (_type_): protein encoder output dimension.
            disease_out_dim (_type_): disease encoder output dimension.
            out_dim (int, optional): output dimension. Defaults to 2.
            drop_out (int, optional): dropout rate. Defaults to 0.
        """
        self.cls = nn.Linear(prot_out_dim + disease_out_dim, out_dim)

    def freeze_encoders(self, freeze_prot_encoder, freeze_disease_encoder):
        """Freeze encoders.

        Args:
            freeze_prot_encoder (boolean): freeze protein encoder
            freeze_disease_encoder (boolean): freeze disease textual encoder
        """
        if freeze_prot_encoder:
            for param in self.prot_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.disease_encoder.parameters():
                param.requires_grad = True
        if freeze_disease_encoder:
            for param in self.disease_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.disease_encoder.parameters():
                param.requires_grad = True
        print(f"freeze_prot_encoder:{freeze_prot_encoder}")
        print(f"freeze_disease_encoder:{freeze_disease_encoder}")
