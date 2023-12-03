import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")


class GDANet(torch.nn.Module):
    def __init__(
        self,
        prot_encoder,
        disease_encoder,
        prot_out_dim=1024,
        disease_out_dim=768,
        drop_out=0,
        freeze_prot_encoder=True,
        freeze_disease_encoder=True,
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
        self.freeze_encoders(freeze_prot_encoder, freeze_disease_encoder)

        self.reg = nn.Sequential(
            nn.Linear(prot_out_dim + disease_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(512, 1),
        )

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

    def forward(self, x1, x2):
        """The forward function.

        Args:
            x1 (tensor): Protein amino acid sequence token ids
            x2 (tensor): Disease textual token ids

        Returns:
            tensor: _description_
        """
        x1 = self.prot_encoder(x1)[0][:, 0]
        x2 = self.disease_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        x = self.reg(x)
        return x


class GDANet_model2(torch.nn.Module):
    def __init__(
        self,
        prot_encoder,
        disease_encoder,
        prot_out_dim=1024,
        disease_out_dim=768,
        drop_out=0,
        freeze_prot_encoder=True,
        freeze_disease_encoder=True,
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
        super(GDANet_model2, self).__init__()
        self.prot_encoder = prot_encoder
        self.disease_encoder = disease_encoder
        self.prot_out_dim = prot_out_dim
        self.disease_out_dim = disease_out_dim
        self.drop_out = drop_out
        self.freeze_encoders(freeze_prot_encoder, freeze_disease_encoder)

        self.reg = nn.Sequential(
            nn.Linear(prot_out_dim + disease_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(512, 1),
        )
        self.add_ppi_head()

    def add_ppi_head(self):
        """Add ppi head for pretraining"""
        self.ppi = nn.Sequential(
            nn.Linear(self.prot_out_dim + self.prot_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(512, 1),
        )

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

    def forward(self, x1, x2):
        """The forward function.

        Args:
            x1 (tensor): Protein amino acid sequence token ids
            x2 (tensor): Disease textual token ids

        Returns:
            tensor: _description_
        """
        x1 = self.prot_encoder(x1)[0][:, 0]
        x2 = self.disease_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        x = self.reg(x)
        return x

    def forward_ppi(self, x1, x2):
        """The forward function.

        Args:
            x1 (tensor): Protein amino acid sequence token ids
            x2 (tensor): Disease textual token ids

        Returns:
            tensor: _description_
        """
        x1 = self.prot_encoder(x1)[0][:, 0]
        x2 = self.prot_encoder(x2)[0][:, 0]
        x = torch.cat((x1, x2), 1)
        x = self.ppi(x)
        return x
