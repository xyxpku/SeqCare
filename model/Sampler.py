import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAE, VGAE, global_add_pool
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, subgraph
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, patient_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels=in_channels, out_channels=hidden_size, add_self_loops=False))
            else:
                self.convs.append(GCNConv(in_channels=hidden_size, out_channels=hidden_size, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.sample_fc_node = nn.Linear(hidden_size+patient_dim,2)
        self.sample_fc_edge = nn.Linear(hidden_size + hidden_size+ patient_dim, 2)
        self.bn_node = nn.BatchNorm1d(2)
        self.bn_edge = nn.BatchNorm1d(2)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.sample_fc_node.weight)
        nn.init.zeros_(self.sample_fc_node.bias)
        nn.init.xavier_uniform_(self.sample_fc_edge.weight)
        nn.init.zeros_(self.sample_fc_edge.bias)

    def forward_node(self, data_input, patient_encoder_output):
        x, edge_index,batch_index  = data_input.x, data_input.edge_index, data_input.batch
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x = self.bns[i](x)
        patient_output_align = torch.index_select(patient_encoder_output, 0, batch_index)
        x = torch.cat((x,patient_output_align),dim=1)
        x_prob = self.sample_fc_node(x)
        x_prob = self.bn_node(x_prob)
        return x_prob

    def forward_edge(self, data_input, patient_encoder_output, edgebindex):
        x, edge_index,batch_index  = data_input.x, data_input.edge_index, data_input.batch
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x = self.bns[i](x)
        src_index = edge_index[0]
        tgt_index = edge_index[1]
        src_node_embed = torch.index_select(x, 0, src_index)
        tgt_node_embed = torch.index_select(x, 0, tgt_index)
        patient_output_align = torch.index_select(patient_encoder_output, 0, edgebindex)
        edge_embed = torch.cat((src_node_embed, tgt_node_embed, patient_output_align), dim=1)
        edge_prob = self.sample_fc_edge(edge_embed)
        edge_prob = self.bn_edge(edge_prob)

        return edge_prob


class Sampler_node(VGAE):
    def __init__(self, in_channels, hidden_size, patient_dim, encoder, num_layers):
        encoder = encoder(in_channels, hidden_size, patient_dim, num_layers)
        super().__init__(encoder=encoder)
    def forward(self, data_input, node_embedding, patient_encoder_output):
        data = copy.deepcopy(data_input)
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        x = node_embedding(x).squeeze()
        data.x = x

        p = self.encoder.forward_node(data,patient_encoder_output)
        sample = F.gumbel_softmax(p, hard=True)

        keep_sample = sample[:, 0]
        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1,)
        edge_index, edge_attr = subgraph(keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr

        return p, keep_sample, data


class Sampler_edge(VGAE):
    def __init__(self, in_channels, hidden_size, patient_dim, encoder, num_layers):
        encoder = encoder(in_channels, hidden_size, patient_dim, num_layers)
        super().__init__(encoder=encoder)

    def filter_adj(self, row, col, edge_attr, mask):
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

    def forward(self, data_input, node_embedding, patient_encoder_output, edgebindex):
        data = copy.deepcopy(data_input)
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr

        x = node_embedding(x).squeeze()
        data.x = x

        p = self.encoder.forward_edge(data, patient_encoder_output, edgebindex)
        sample = F.gumbel_softmax(p, hard=True)
        mask = sample[:, 0] >= 1
        row, col = edge_index
        row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)
        edge_index = torch.stack([row, col], dim=0)
        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr

        return p, sample[:,0], data
