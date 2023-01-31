import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class attentionGRU(nn.Module):
    def __init__(self, dim_emb=64, dim_hidden=128, gru_dropout_prob =0.,
                 batch_first=True):
        super(attentionGRU, self).__init__()
        self.batch_first = batch_first
        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_hidden, num_layers=1, dropout=gru_dropout_prob, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_hidden, out_features=1)

    def _init_weight(self):
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

    def forward(self, x, lengths, batch_mask, device):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)

        g, _ = self.rnn_alpha(packed_input)

        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first,total_length=max_len)

        batch_mask = batch_mask.unsqueeze(2)
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp


        alpha = masked_softmax(e, batch_mask)

        output = torch.bmm(torch.transpose(alpha, 1, 2), x).squeeze(1)

        return alpha, output