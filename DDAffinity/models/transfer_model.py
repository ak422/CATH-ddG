#!/user/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from typing import Optional
from DDAffinity.models.protein_mpnn_utils import ProteinMPNN, tied_featurize, alt_parse_PDB


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True



@dataclass
class Mutation:
    position: int
    wildtype: str
    chain:str
    mutation: str
    ddG: Optional[float] = None
    pdb: Optional[str] = ''


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3

    # model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(cfg.vanilla_model_weight, version)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                        k_neighbors=16, augment_eps=0.2)
    if cfg.vanilla_model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])

    if cfg.vanilla_model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model


class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.vanilla_model.hidden_dims)
        self.subtract_mut = cfg.vanilla_model.subtract_mut
        self.num_final_layers = cfg.vanilla_model.num_final_layers
        self.lightattn = cfg.vanilla_model.lightattn if 'lightattn' in cfg.vanilla_model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'

        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128
        HIDDEN_DIM = 128

        hid_sizes = [HIDDEN_DIM * self.num_final_layers + EMBED_DIM]
        hid_sizes += self.hidden_dims
        hid_sizes += [VOCAB_DIM]

        # print('MLP HIDDEN SIZES:', hid_sizes)

        if self.lightattn:
            # print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=HIDDEN_DIM * self.num_final_layers + EMBED_DIM)

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(HIDDEN_DIM*3, HIDDEN_DIM)

    # def forward(self, pdb, mutations, tied_feat=True):
    def forward(self, batch, tied_feat=True):
        X = batch['pos_heavyatom']
        S = (batch['aa'] * batch['mask']).long()
        mask = batch['mask'].float()
        chain_M = mask
        residue_idx = batch['residue_idx']
        chain_encoding_all = batch['chain_nb']

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)

        mut_embeddings = []
        # for mut in final_mutation_list:
        inputs = []
        if self.num_final_layers > 0:
            hid = mpnn_hid  # MPNN hidden embeddings at mut position
            inputs.append(hid)

        embed = mpnn_embed  # MPNN seq embeddings at mut position
        inputs.append(embed)

        # concatenating hidden layers and embeddings
        lin_input = torch.cat(inputs, -1)

        # passing vector through lightattn
        if self.lightattn:
            lin_input = self.light_attention(lin_input, mask)
        lin_input = self.ddg_out(lin_input)
        return lin_input


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """

    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x.transpose(1,2))  # [batch_size, embeddings_dim, sequence_length]

        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x.transpose(1,2))  # [batch_size, embeddings_dim, sequence_length]

        o1 = o * self.softmax(attention)
        return o1.transpose(1,2)



class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Qdim, Mdim)

    def forward(self, Q, K, V, mask=None):
        Q, K = self.WQ(Q), self.WK(K)
        Attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.model_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask.unsqueeze(1), -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        return torch.matmul(Attn, V)
