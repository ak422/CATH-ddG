# -*- coding: utf-8 -*-
from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
from collections.abc import Sequence
import math
import pandas as pd
import copy

PI = math.pi
import torch
import random
from torch import optim
from torch.utils.data import DataLoader
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from DDAffinity.modules.encoders.single import PerResidueEncoder, AAEmbedding
from DDAffinity.utils.protein.dihedral_chi import CHI_PI_PERIODIC_LIST
from copy import deepcopy

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.relpos_embed = nn.Embedding(2 * max_relative_feature + 1, num_embeddings)
        self.chains_embed = nn.Embedding(2, num_embeddings)

    def forward(self, offset, chains):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        E = self.relpos_embed(d) + self.chains_embed(chains)
        return E

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def gaussian(x, mean, std):
    pi = 3.1415926
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

# class NonLinear(nn.Module):
#     def __init__(self, input, output_size, hidden=None):
#         super(NonLinear, self).__init__()
#
#         if hidden is None:
#             hidden = input
#         self.layer1 = nn.Linear(input, hidden)
#         self.layer2 = nn.Linear(hidden, output_size)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = F.gelu(x)
#         x = self.layer2(x)
#         return x

class Gaussian_neighbor(nn.Module):
    def __init__(self, K=16, edge_types=5*5):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)  # 维度 = K
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)        # 维度 = 1
        self.bias = nn.Embedding(edge_types, 1)       # padding_idx: 标记输入中的填充值
    def gather_nodes(self, nodes, neighbor_idx):
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features
    def forward(self, X, E_idx, mask_atoms, mask_attend):
        n_complex, n_residue, n_atom, _ = X.shape
        X_views = X.view((X.shape[0], X.shape[1], -1))
        neighbors_X = self.gather_nodes(X_views, E_idx)
        neighbors_X = neighbors_X.view(X.shape[0], X.shape[1], -1, 5, 3)
        delta_pos = (neighbors_X.unsqueeze(-2) - X.unsqueeze(2).unsqueeze(-3)).norm(dim=-1)  #　邻居节点到Ｘ所有原子的距离

        D_A_B_neighbors = delta_pos.to(X.device)
        edge_types = torch.arange(0, n_atom*n_atom).view(n_atom, n_atom).to(X.device)
        mul = self.mul(edge_types).squeeze(-1)   # 边类型嵌入，然后对边求和， gamma
        bias = self.bias(edge_types).squeeze(-1)    # 边类型嵌入，然后对边求和, beta

        x = mul * D_A_B_neighbors+ bias
        mask_atoms_attend = self.gather_nodes(mask_atoms, E_idx)
        mask_attend_new = (mask_atoms_attend[:, :, :, None, :] * mask_atoms_attend[:, :, :, :, None])
        x = x * mask_attend_new
        x = x.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        gbf = gaussian(x.float(), mean, std).type_as(self.means.weight).view(n_complex, n_residue, -1,n_atom*n_atom*self.K)
        return gbf * mask_attend.unsqueeze(-1)

class AApair(nn.Module):
    def __init__(self, K=16, max_aa_types=22):
        super().__init__()
        self.K = K
        self.max_aa_types = max_aa_types
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, self.K, padding_idx=21)
    def forward(self, aa, E_idx, mask_attend):
        # Pair identities[氨基酸类型pair编码]
        aa_pair = ((aa[:, :, None] + 1) % self.max_aa_types) * self.max_aa_types + \
                  ((aa[:, None, :] + 1) % self.max_aa_types)
        aa_pair = torch.clamp(aa_pair, min=21)
        aa_pair = torch.where(aa_pair % self.max_aa_types == 0, 21, aa_pair)

        aa_pair_neighbor = torch.gather(aa_pair, 2, E_idx)
        feat_aapair = self.aa_pair_embed(aa_pair_neighbor.to(torch.long))

        return feat_aapair * mask_attend.unsqueeze(-1)

class ResiduePairEncoder(nn.Module):
    def __init__(self, edge_features, node_features, top_k1, top_k2, k3, long_range_seq, noise_bb, noise_sd, num_positional_embeddings=16, num_rbf=16):
        """ Extract protein features """
        super(ResiduePairEncoder, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k1 = top_k1
        self.top_k2 = top_k2
        self.k3 = k3
        # self.top_k4 = top_k4
        self.long_range_seq = long_range_seq
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.noise_bb = noise_bb
        self.noise_sd = noise_sd

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25 + 7 * 2 +16
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 45 + 7 * 2 + 16
        self.aapair = AApair(K=16, max_aa_types=22)

        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features, elementwise_affine=False)

        self.dropout = nn.Dropout(0.1)

    def _dist(self, X, mask_residue, residue_idx, cutoff=15.0, eps=1E-6):
        """ Pairwise euclidean distances """
        B, N = X.size(0), X.size(1)
        mask_2D = torch.unsqueeze(mask_residue, 1) * torch.unsqueeze(mask_residue, 2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + eps) * mask_2D
        # mask_rball = (D < cutoff) * mask_2D
        mask_rball = mask_2D
        D_max, _ = torch.max(D, -1, keepdim=True)

        # 序列最近邻
        rel_residue = residue_idx[:, :, None] - residue_idx[:, None, :]
        mask_2D_seq = (torch.abs(rel_residue) <= (self.k3 - 1) / 2) * mask_rball
        D_sequence = D * mask_2D_seq  # 以距离对齐
        _, E_idx_seq = torch.topk(D_sequence, self.k3, dim=-1, largest=True)
        mask_seq = gather_edges(mask_2D_seq.unsqueeze(-1), E_idx_seq)[:, :, :, 0]

        # 空间最近邻
        # Identify k nearest neighbors (including self)
        D_adjust = D + (~mask_rball) * D_max
        _, E_idx_spatial = torch.topk(D_adjust, self.top_k1, dim=-1, largest=False)  # 取最小值
        mask_spatial = gather_edges(mask_rball.unsqueeze(-1), E_idx_spatial)[:, :, :, 0]

        # 序列远 空间近
        mask_2D_seq = (torch.abs(rel_residue) <= (self.long_range_seq - 1) / 2) * mask_rball  # masked sequence
        mask_2D_Lrange = (~mask_2D_seq) * mask_rball  # 序列远
        D_adjust = D + (~mask_2D_Lrange) * D_max
        _, E_idx_Lrange = torch.topk(D_adjust, self.top_k2, dim=-1, largest=False)  # 取最小值
        mask_Lrange = gather_edges(mask_2D_Lrange.unsqueeze(-1), E_idx_Lrange)[:, :, :, 0]

        return E_idx_spatial, mask_spatial, E_idx_seq, mask_seq, E_idx_Lrange, mask_Lrange

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q
    def _orientations_coarse(self, X, E_idx, mask_attend):
        # Pair features
        u = torch.ones_like(X)
        u[:,1:,:] = X[:, 1:, :] - X[:,:-1,:]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(X)
        b[:, :-1,:] = u[:, :-1,:] - u[:, 1:,:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(X)
        n[:,:-1,:] = torch.cross(u[:,:-1,:], u[:,1:,:], dim=-1)
        n = F.normalize(n, dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=2)
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [9])

        X_neighbors = gather_nodes(X, E_idx)
        O_neighbors = gather_nodes(local_frame, E_idx)
        # Re-view as rotation matrices
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [3, 3])    # Oi
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])    # Oj
        # # # Rotate into local reference frames ，计算最近邻相对x_i的局部坐标系
        t = X_neighbors - X.unsqueeze(-2)
        t = torch.matmul(local_frame.unsqueeze(2), t.unsqueeze(-1)).squeeze(-1)  # 边特征第二项
        t = F.normalize(t, dim=-1) * mask_attend.unsqueeze(-1)
        r = torch.matmul(local_frame.unsqueeze(2).transpose(-1, -2), O_neighbors)  # 边特征第三项
        r = self._quaternions(r)  * mask_attend.unsqueeze(-1)   # 边特征第三项
        t2 = (1 - 2 * t) * mask_attend.unsqueeze(-1)
        r2 = (1 - 2 * r) * mask_attend.unsqueeze(-1)

        return torch.cat([t, r, t2, r2], dim=-1)

    # def PerEdgeEncoder(self, X, E_idx, mask_attend, residue_idx, chain_labels):
    def PerEdgeEncoder(self, X, E_idx, mask_attend, residue_idx, chain_labels):
        # 1. Relative spatial encodings
        O_features = self._orientations_coarse(X, E_idx, mask_attend)

        # 2. 相对位置编码
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]
        # d_chains：链内和链间标记，为0表示链内
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]

        E_positional = self.embeddings(offset.long(), E_chains)

        return E_positional, O_features, offset.long()

    def _set_Cb_positions(self, X, mask_atom):
        """
        Args:
            pos_atoms:  (L, A, 3)
            mask_atoms: (L, A)
        """
        # X[:, :, 0:4] = X[:, :, 0:4] + self.noise_bb * torch.randn_like(X[:, :, 0:4])
        # X[:, :, 4:] = X[:, :, 4:] + self.noise_sd * torch.randn_like(X[:, :, 4:])

        Ca = X[:, :, 1]
        b = X[:, :, 1] - X[:, :, 0]
        c = X[:, :, 2] - X[:, :, 1]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1]  # 虚拟Cb原子
        X[:, :, 4] = torch.where(mask_atom[:, :, 4, None], X[:, :, 4], Cb)

        X[:, :, 0:4] = X[:, :, 0:4] + self.noise_bb * torch.randn_like(X[:, :, 0:4])
        X[:, :, 4:] = X[:, :, 4:] + self.noise_sd * torch.randn_like(X[:, :, 4:])
        return X

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF
    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B
    # def _rbf_residue(self, D, num_rbf=4):
    #     device = D.device
    #     D_min, D_max, D_count = 2., 22., num_rbf
    #     D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    #     D_mu = D_mu.view([1,1,-1])
    #     D_sigma = (D_max - D_min) / D_count
    #     D_expand = torch.unsqueeze(D, -1)
    #     RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    #     return RBF
    # def _get_rbf_residue(self, A, B):
    #     D_A_B = torch.sqrt(torch.sum((A[:,:,:] - B[:,:,:])**2,-1) + 1e-6) #[B, L, L]
    #     RBF_A_B = self._rbf_residue(D_A_B)
    #     return RBF_A_B

    def forward(self, batch):
        mask_atom = batch["mask_atoms"]   # N = 0; CA = 1; C = 2; O = 3; CB = 4;
        residue_idx = batch["residue_idx"]
        res_nb = batch["res_nb"]
        chain_labels = batch["chain_nb"]     # # d_chains：链内和链间标记，为1表示链内
        mask_residue = batch["mask"]
        aa = batch["aa"]

        batch["pos_heavyatom"] = self._set_Cb_positions(batch["pos_heavyatom_ori"], mask_atom)
        # batch["pos_heavyatom"] = self._set_Cb_positions(batch["pos_heavyatom"], mask_atom)
        X = batch["pos_heavyatom"]
        N = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        Cb = X[:, :, 4, :]

        # 这里考虑用Cb原子来计算最短距离
        E_idx_spatial, mask_spatial, E_idx_seq, mask_seq, E_idx_Lrange, mask_Lrange = self._dist(Cb, mask_residue, residue_idx)

        # spactial coding & sequential coding
        E_idx = torch.cat([E_idx_spatial, E_idx_Lrange, E_idx_seq], dim=-1)
        mask_attend = torch.cat([mask_spatial, mask_Lrange, mask_seq], dim=-1)
        mask_attend = mask_residue.unsqueeze(-1) * mask_attend
        E_positional, O_features, offset = self.PerEdgeEncoder(Cb, E_idx, mask_attend, residue_idx, chain_labels)

        # backbone rbf
        RBF_all = []
        RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        # RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # sidechain rbf
        mask_heavyatom = batch["mask_heavyatom"]
        for i in range(5, 15, 1):
            mask_i = mask_heavyatom[:, :, i]
            mask_j = torch.gather(mask_i.unsqueeze(-1).expand(-1, -1, mask_heavyatom.size(1)), 2, E_idx)
            # Cb-atoms
            RBF_all.append(self._get_rbf(Cb, X[:, :, i, :], E_idx) * mask_j[:, :, :, None])

        for i in range(5, 15, 1):
            mask_i = mask_heavyatom[:, :, i]
            # atoms-Cb
            RBF_all.append(self._get_rbf(X[:, :, i, :], Cb, E_idx) * mask_i[:, :, None, None])
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        aapair_feature = self.aapair(aa, E_idx, mask_attend)

        E = torch.cat((E_positional, RBF_all, O_features, aapair_feature), dim=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        idx_spatial = self.top_k1 + self.top_k2
        idx_seq = self.k3

        E_spatial = E[...,:idx_spatial,:]
        E_idx_spatial = E_idx[...,:idx_spatial]
        mask_spatial = mask_attend[...,:idx_spatial]
        E_seq = E[..., -idx_seq:, :]
        E_idx_seq = E_idx[..., -idx_seq:]
        mask_seq = mask_attend[..., -idx_seq:]

        return E_spatial, E_idx_spatial, mask_spatial, E_seq, E_idx_seq, mask_seq

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden, elementwise_affine=False) for i in range(6)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        # ReZero is All You Need: Fast Convergence at Large Depth
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.act = torch.nn.SiLU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(0, h_V, before=True, after=False)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)         # h_j(enc) || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)   # h_i(enc)
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # h_i(enc) || h_j(enc) || edge
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        # ReZero
        dh = dh * self.resweight
        h_V = residual + self.dropout1(dh)

        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(2, h_V, before=True, after=False)
        # Position-wise feedforward
        dh = self.dense(h_V)
        # ReZero
        dh = dh * self.resweight
        h_V = residual + self.dropout2(dh)
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # pre-norm
        residual = h_E
        h_E = self.maybe_layer_norm(4, h_E, before=True, after=False)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # h_j || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)  # h_i
        h_EV = torch.cat([h_V_expand, h_EV], -1)   # h_i || h_j || edge
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = residual + self.dropout3(h_message)

        return h_V, h_E

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        return self.layer_norms[i](x)

class FusionLayer(nn.Module):
    def __init__(self, num_in, num_hidden,  normalize_before=False, dropout=0.1, num_heads=None, scale=30):
        super(FusionLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden, elementwise_affine=False) for i in range(3)])

        self.normalize_before = normalize_before

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.SiLU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_S, h_V, h_E, E_idx, mask_attend, mask_V=None):
        """ Parallel computation of full transformer layer """
        # Concatenate h_V_i to h_E_ij
        residual = h_V
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # h_j || h_edge
        h_EXV_out = cat_neighbors_nodes(h_V, h_ES, E_idx)  # h_j || h_j(enc) || h_edge
        h_EXV_out = mask_attend.unsqueeze(-1) * h_EXV_out

        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_S_expand = h_S.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_S_expand, h_V_expand, h_EXV_out], -1)  # h_i || h_i(enc) ||  h_j(enc) || h_j || h_edge
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = residual + self.dropout1(dh)
        h_V = self.maybe_layer_norm(2, h_V, before=False, after=True)
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        return self.layer_norms[i](x)

def init_params(module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        if data.dim() > 1:
            nn.init.xavier_uniform_(data)

    if isinstance(module, Gaussian_neighbor):
        nn.init.uniform_(module.means.weight, 0, 3)
        nn.init.uniform_(module.stds.weight, 0, 3)
        nn.init.constant_(module.bias.weight, 0)
        nn.init.constant_(module.mul.weight, 1)
    elif isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    # elif isinstance(module, nn.LayerNorm):
    #     # 初始化层归一化的权重为 1，偏置为 0
    #     module.bias.data.zero_()
    #     module.weight.data.fill_(1.0)

# class Adapter(nn.Module):
#     def __init__(self, input, output_size, hidden=128):
#         super(Adapter,self).__init__()
#         self.n_embd = input
#         self.up_size = hidden
#
#         #_before
#         # self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
#
#         # if adapter_scalar == "learnable_scalar":
#         #     self.scale = nn.Parameter(torch.ones(1))
#         # else:
#         #     self.scale = float(adapter_scalar)
#
#         self.down_proj = nn.Linear(self.n_embd, self.up_size)
#         self.non_linear_func = nn.GELU()
#         self.up_proj = nn.Linear(self.up_size, self.n_embd)
#         self.output_proj = nn.Linear(self.n_embd, output_size)
#
#         # with torch.no_grad():
#         #     nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#         #     nn.init.zeros_(self.up_proj.weight)
#         #     nn.init.zeros_(self.down_proj.bias)
#         #     nn.init.zeros_(self.up_proj.bias)
#         #     nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))
#         #     nn.init.zeros_(self.output_proj.bias)
#     def forward(self, x, add_residual=True, residual=None):
#         residual = x
#         # x = self.adapter_layer_norm_before(x)
#
#         down = self.down_proj(x)
#         down = self.non_linear_func(down)
#         # down = nn.functional.dropout(down, p=0.1)
#         up = self.up_proj(down)
#
#         if add_residual:
#             output = up + residual
#         else:
#             output = up
#         output = self.output_proj(output)
#
#         return output

class DDAffinity_NET(nn.Module):
    def __init__(self, cfg):
        super(DDAffinity_NET, self).__init__()
        # Hyperparameters
        self.node_features = cfg.encoder.node_feat_dim
        self.edge_features = cfg.encoder.edge_feat_dim
        self.num_encoder_layers = cfg.encoder.num_layers
        self.dropout = cfg.dropout
        hidden_dim = cfg.hidden_dim
        self.num_rbf = 16
        self.top_k1 = cfg.k1
        self.top_k2 = cfg.k2
        self.k3 = cfg.k3
        self.long_range_seq = cfg.long_range_seq

        self.pair_encoder = ResiduePairEncoder(self.edge_features, self.node_features, top_k1=cfg.k1, top_k2= cfg.k2, k3 = cfg.k3, long_range_seq=cfg.long_range_seq, noise_bb=cfg.noise_bb, noise_sd=cfg.noise_sd)
        self.W_e = nn.Linear(self.edge_features, hidden_dim, bias=True)
        self.W_es = nn.Linear(self.edge_features, hidden_dim, bias=True)

        # Residue Encoding  # N, CA, C, O, CB,
        self.single_encoders = nn.ModuleList([
            PerResidueEncoder(
                feat_dim=cfg.encoder.node_feat_dim,
            )
            for _ in range(2)
        ])

        self.AA_embed = nn.ModuleList([
            AAEmbedding(feat_dim=cfg.encoder.node_feat_dim, infeat_dim=123)
            for _ in range(2)
        ])

        self.binding_embed = nn.ModuleList([
            nn.Embedding(
                        num_embeddings=2,
                        embedding_dim=cfg.encoder.node_feat_dim,
                        padding_idx=0,
                         )
            for _ in range(2)
        ])

        self.mut_embed = nn.ModuleList([
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=cfg.encoder.node_feat_dim,
                padding_idx=0,
            )
            for _ in range(2)
        ])

        # Encoder layers
        self.encoder_layers_spatial = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=cfg.dropout)
            for _ in range(self.num_encoder_layers)
        ])
        self.encoder_layers_sequential = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=cfg.dropout)
            for _ in range(self.num_encoder_layers)
        ])

        self.single_fusion = nn.Linear(hidden_dim*2, hidden_dim)

        self.fusion_layer = FusionLayer(hidden_dim * 4, hidden_dim, dropout=cfg.dropout)
        self.enc_centrality = nn.Parameter(torch.Tensor([0]))

        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),   # nn.Tanh(), nn.GELU()
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),   # nn.Tanh(), nn.GELU()
            nn.Linear(hidden_dim, 1)
        )

        # foldx_ddg
        self.foldx_ddg = nn.Sequential(
            nn.Linear(15, hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, 1)
        )

        # cath classifier
        self.cath_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden_dim, cfg.num_labels)
            )

        self.BCEWithLogLoss = nn.BCEWithLogitsLoss()

        self.apply(lambda module: init_params(module))

    def gather_centrality(self, nodes, neighbor_idx, mask):
        # Fe2atures [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:2] + [-1])
        # # 　增加可学习参数
        neighbor_features = neighbor_features + self.enc_centrality.unsqueeze(0)
        centrality_norm = torch.nn.functional.normalize(torch.sum(neighbor_features, dtype=torch.float16,dim=-1), dim=-1) * mask

        return centrality_norm[:, :, None]

    def _random_flip_chi(self, chi, chi_alt):
        """
        Args:
            chi: (L, 4)
            flip_prob: float
            chi_alt: (L, 4)
        """
        chi_new = torch.where(
            torch.rand_like(chi) <= 0.2,
            chi_alt,
            chi,
        )
        return chi_new

    def dihedral_encode(self, batch, code_idx):
        mask_residue = batch['mask']

        chi = self._random_flip_chi(batch['chi'], batch['chi_alt'])
        # chi = batch['chi']
        chi_select = chi * (1 - batch['mut_flag'].float())[:, :, None]

        x = self.single_encoders[code_idx](
            aa=batch['aa'],
            aa_esm2=batch['aa_esm2'],
            # X=batch["pos_atoms"],
            X=batch["pos_heavyatom"], mask_atom=batch['mask_heavyatom'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi_select, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue
        )

        # 氨基酸极性编码
        aa_embed = self.AA_embed[code_idx](batch['aa'], mask_residue)

        # binding chains
        b = self.binding_embed[code_idx](batch["is_binding"])
        m = self.mut_embed[code_idx](batch['mut_flag'].long())

        x = x + aa_embed + b + m  # (6,128,128)

        return x

    def encode(self, batch):
        # 编码器
        E_spatial, E_idx_spatial, mask_spatial, E_seq, E_idx_seq, mask_seq= self.pair_encoder(batch)
        h_E_spatial = self.W_e(E_spatial)
        h_E_seq = self.W_es(E_seq)

        mask = batch['mask']
        # Encoder
        h_V_spatial = self.dihedral_encode(batch, 0)
        for i, layer in enumerate(self.encoder_layers_spatial):
            h_V_spatial, h_E_spatial = layer(h_V_spatial, h_E_spatial, E_idx_spatial, mask, mask_spatial)

        h_V_sequential = h_V_spatial
        for i, layer in enumerate(self.encoder_layers_sequential):
            h_V_sequential, h_E_seq = layer(h_V_sequential, h_E_seq, E_idx_seq, mask, mask_seq)

        h_S = self.dihedral_encode(batch, 1)
        h_V = self.single_fusion(torch.cat([h_V_spatial, h_V_sequential],dim=-1))
        E_idx = torch.cat([E_idx_spatial, E_idx_seq], dim=-1)
        h_E = torch.cat([h_E_spatial, h_E_seq], dim=-2)
        mask_attend = torch.cat([mask_spatial, mask_seq], dim=-1)

        h_EXV_out = self.fusion_layer(h_S, h_V, h_E, E_idx, mask_attend, mask)

        # # 中心性
        h_centrality = self.gather_centrality(batch["centrality"], E_idx, mask)
        h_EXV_out = h_EXV_out * h_centrality

        return h_EXV_out

    def forward(self, batch):
        """
        Graph-conditioned sequence model
        """
        batch_wt = batch["wt"]
        batch_mt = batch["mt"]
        B, L = batch_wt['aa'].size()
        is_single = torch.where(batch_wt["num_muts"] > 1, 0, 1)[:, None]  # True: single, False: multiple
        num_is_single = torch.clamp(torch.sum(batch_wt["num_muts"] == 1), 1)
        num_is_multi = torch.clamp(torch.sum(batch_wt["num_muts"] > 1), 1)

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)
        h_wt = h_wt * batch_wt['mut_flag'][:, :, None]
        h_mt = h_mt * batch_mt['mut_flag'][:, :, None]

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        # # save latent
        # df_wt = pd.DataFrame(H_wt.cpu().detach().numpy(),
        #                      index=np.array(batch_wt['pdbcode']))
        # df_mt = pd.DataFrame(H_mt.cpu().detach().numpy(),
        #                      index=np.array(batch_mt['pdbcode']))
        # df_wt.to_csv('df_wt_hidden.csv', mode='a', index=True, header=None)
        # df_mt.to_csv('df_mt_hidden.csv', mode='a', index=True, header=None)

        ddg_pred_foldx = self.foldx_ddg(batch_mt['inter_energy'] - batch_wt['inter_energy'])
        ddg_pred_foldx_inv = self.foldx_ddg(batch_wt['inter_energy'] - batch_mt['inter_energy'])
        loss_foldx = (F.mse_loss(ddg_pred_foldx, batch['ddG']) + F.mse_loss(ddg_pred_foldx_inv, -batch['ddG']))/2

        ddg_pred = self.ddg_readout(H_mt - H_wt)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt)
        loss_single = (F.mse_loss(ddg_pred * is_single, batch['ddG'] * is_single, reduction="sum") + F.mse_loss(ddg_pred_inv * is_single, -batch['ddG'] * is_single, reduction="sum")) / (2 * num_is_single)
        loss_multi = (F.mse_loss(ddg_pred * (1 - is_single), batch['ddG'] * (1 - is_single), reduction="sum") + F.mse_loss(ddg_pred_inv * (1 - is_single), -batch['ddG'] * (1 - is_single), reduction="sum")) / (2 * num_is_multi)
        loss_mse = 0.6 * loss_single + 0.4 * loss_multi

        # loss_mse = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        # cath domain classifier
        logits_wt = self.cath_classifier(H_wt) # {0:0,1:1,2:2,3:3,4:4,6:5}
        logits_mt = self.cath_classifier(H_mt)
        loss_cath = (self.BCEWithLogLoss(input=logits_wt, target=batch_wt['cath_domain']) + \
                    self.BCEWithLogLoss(input=logits_mt, target=batch_mt['cath_domain']))/2

        loss_stack = {
            'loss_mse': loss_mse,
            'loss_foldx': loss_foldx,
            'loss_cath': loss_cath,
        }

        return loss_stack
    def inference(self, batch):
        """
        Graph-conditioned sequence model
        """
        batch_wt = batch["wt"]
        batch_mt = batch["mt"]

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)
        h_wt = h_wt * batch_wt['mut_flag'][:, :, None]
        h_mt = h_mt * batch_mt['mut_flag'][:, :, None]

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        ddg_pred_foldx = self.foldx_ddg(batch_mt['inter_energy'] - batch_wt['inter_energy'])
        ddg_pred = 0.5 * self.ddg_readout(H_mt - H_wt) + 0.5 * ddg_pred_foldx

        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }
        return out_dict
