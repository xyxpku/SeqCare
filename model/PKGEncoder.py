from model.CAGATConv import CAGATConv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter
import numpy as np
from torch_geometric.data import Data
import math
from torch_geometric.loader import DataLoader

class PKGEncoder(nn.Module):
    def __init__(self, gencoder_dim_list, gencoder_mess_dropout, n_layers_gencoder , edge_dim,
                  global_pool="mean",graph_encoder_choice = "CAGAT"):
        super().__init__()
        self.gencoder_dim_list = gencoder_dim_list
        self.gencoder_mess_dropout = gencoder_mess_dropout
        self.n_layers_gencoder = n_layers_gencoder
        self.edge_dim = edge_dim
        self.graph_encoder_choice =graph_encoder_choice
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool

        self.relu = nn.ReLU()
        self.bns = nn.ModuleList()
        self.gencoders = nn.ModuleList()
        for i in range(self.n_layers_gencoder):
            self.bns.append(nn.BatchNorm1d(self.gencoder_dim_list[i]))
            if self.graph_encoder_choice == "CAGAT":
                self.gencoders.append(CAGATConv(in_channels=self.gencoder_dim_list[i],out_channels=self.gencoder_dim_list[i+1],
                                            edge_dim = self.edge_dim, dropout= self.gencoder_mess_dropout[i]))

        hidden_size = np.sum(np.array(self.gencoder_dim_list))
        self.bn_hidden = nn.BatchNorm1d(hidden_size)
        self.proj_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size))

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

    def forward_gcl(self,data_input,  W_r, relation_embedding, mode ="pre-training"):
        x, edge_index, batch, edge_attr = data_input.x, data_input.edge_index, data_input.batch, data_input.edge_attr
        all_embed = [x]

        for i, gencoder in enumerate(self.gencoders):
            x = self.bns[i](x)


            x = self.relu(
                gencoder(x, edge_index, edge_attr, W_r, relation_embedding, mode=mode))
            all_embed.append(x)
        all_embed = torch.cat(all_embed, dim=1)

        all_embed = self.global_pool(all_embed, batch)
        all_embed = self.bn_hidden(all_embed)
        all_embed = self.proj_head(all_embed)

        return all_embed

    def forward_logits(self,data_input,  W_r, relation_embedding, mode ="train"):
        x, edge_index, batch, edge_attr = data_input.x, data_input.edge_index, data_input.batch, data_input.edge_attr
        all_embed = [x]

        for i, gencoder in enumerate(self.gencoders):
            x = self.bns[i](x)
            x = self.relu(
                gencoder(x, edge_index, edge_attr, W_r, relation_embedding, mode=mode))
            all_embed.append(x)
        all_embed = torch.cat(all_embed, dim=1)
        all_embed = self.global_pool(all_embed, batch)
        return all_embed

