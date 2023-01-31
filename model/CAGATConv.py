import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax  
from torch_scatter import scatter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import Data
import math
from torch_geometric.loader import DataLoader

class CAGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_dim,
                 negative_slope=0.2,
                 dropout=0.,
                 add_self_loops=False,
                 bias=False):
        super(CAGATConv, self).__init__(node_dim=0,flow="target_to_source", aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.W_aggr = Linear(in_channels, out_channels, weight_initializer='glorot')
        self.W_att = nn.Parameter(torch.Tensor(1, edge_dim))
        self.tanh = nn.Tanh()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_aggr.reset_parameters()
        nn.init.xavier_uniform_(self.W_att, gain=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, W_r, relation_embedding, size=None, return_attention_weights=None, mode ="pre-training"):
        if self.add_self_loops == True:
            if isinstance(edge_index, nn.Tensor):
                num_nodes = x.size(0)
                if x is not None:
                    num_nodes = min(num_nodes, x.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=0,
                    num_nodes=num_nodes)
                
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, W_r=W_r, relation_embedding= relation_embedding)
        out = self.activation(self.W_aggr(out + x))
        if self.bias is not None:
            out += self.bias
        out = self.message_dropout(out)

        if isinstance(return_attention_weights, bool):
            return out
        else:
            return out

    def message(self, x_i, x_j, edge_attr, W_r, relation_embedding, index, ptr,
                    size_i):
        W_r_node = torch.index_select(W_r, 0, edge_attr)
        edge_embedding = relation_embedding(edge_attr)         
        x_i_edge = torch.einsum('ik,ikl->il', [x_i,W_r_node])   
        x_j_edge = torch.einsum('ik,ikl->il', [x_j, W_r_node])
        edge_information = self.tanh((x_i_edge + x_j_edge + edge_embedding).view(-1,
                                                                       self.edge_dim))
        
        alpha = (self.W_att * edge_information).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha.unsqueeze(-1) * x_j


        
        
        

