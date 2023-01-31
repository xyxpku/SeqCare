import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CosineClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, temperature_init_value, cooccur_gnn_layers, know_mask_rate):
        super(CosineClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cosine_linear = nn.Linear(in_channels, num_classes, bias=False)
        self.temperature = nn.Parameter(torch.FloatTensor(1))
        self.temperature_init_value = temperature_init_value
        self.cooccur_gnn_layers = cooccur_gnn_layers
        self.know_mask_rate = know_mask_rate

        self.cooccur_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.cooccur_gnn_layers):
            self.cooccur_convs.append(GCNConv(in_channels=in_channels, out_channels=in_channels, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(in_channels))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.cosine_linear.weight)
        nn.init.constant_(self.temperature, self.temperature_init_value)

    def filter_adj(self, row, col, edge_attr, mask):
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

    def forward(self, input, cooccur_matrix):
        assert input.dim() == 2


        x, edge_index, edge_weight = cooccur_matrix.x, cooccur_matrix.edge_index, cooccur_matrix.edge_attr
        if self.know_mask_rate != 0.0:
            row, col = edge_index
            mask = torch.rand(row.size(0), device=edge_index.device) >= self.know_mask_rate
            mask[row > col] = False
            row, col, edge_weight = self.filter_adj(row, col, edge_weight, mask)
            edge_index = torch.stack([row, col], dim=0)

        cosine_weight = self.cosine_linear.weight
        x = torch.index_select(cosine_weight, 0, x.squeeze())

        for i, conv in enumerate(self.cooccur_convs):
            x = F.relu(conv(x, edge_index, edge_weight))
            x = self.bns[i](x)


        weight_norm = F.normalize(x, p=2, dim=1)
        input_norm = F.normalize(input, p=2, dim=1)
        output = self.temperature * torch.einsum('ik,jk->ij', [input_norm,weight_norm])
        return output

