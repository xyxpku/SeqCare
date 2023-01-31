import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from model.attentionGRU import attentionGRU
from model.PKGEncoder import PKGEncoder
from model.Sampler import Sampler_node,Sampler_edge, GCN_Encoder
from model.cosine_classifier import CosineClassifier

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class MLP(nn.Module):
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh , 'leakyrelu':nn.LeakyReLU}

    def __init__(self, input_size, hidden_size_list, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='leakyrelu',use_dropout = False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):


            n_in = self.input_size if i == 0 else self.hidden_size_list[i-1]
            n_out = self.hidden_size_list[i] if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                if use_dropout:
                    self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)


        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.layers.apply(init_weight)

    def forward(self, input):
        return self.layers(input)

class SeqCare(nn.Module):
    def __init__(self, args,
                 node_num, code_num,relation_num, max_visit_len
                 ):
        super(SeqCare, self).__init__()
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.node_num = node_num
        self.relation_num = relation_num
        self.code_num = code_num

        self.max_visit_len = max_visit_len
        print("code_num：", self.code_num)
        print("max_visit_len：",self.max_visit_len)

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.gencoder_dim_list = [args.entity_dim] + eval(args.gencoder_dim_list)
        self.gencoder_mess_dropout = [args.gencoder_mess_dropout] * len(eval(args.gencoder_dim_list))
        self.n_layers_gencoder = len(eval(args.gencoder_dim_list))


        self.gru_dropout_rate = args.gru_dropout_prob
        self.gru_layer = args.gru_layer
        self.gru_hidden_size = args.gru_hidden_size
        self.gru_input_size = args.code_dim
        self.patient_dim = self.gru_input_size


        self.gru = attentionGRU(dim_emb=self.gru_input_size, dim_hidden=self.gru_hidden_size, gru_dropout_prob =self.gru_dropout_rate)

        self.sampler_gnn_layers = args.sampler_gnn_layers

        self.sampler_node = Sampler_node(self.entity_dim, self.entity_dim, self.patient_dim, GCN_Encoder,
                                         self.sampler_gnn_layers)
        self.sampler_edge = Sampler_edge(self.entity_dim, self.entity_dim, self.patient_dim, GCN_Encoder,
                                         self.sampler_gnn_layers)

        self.global_pool = args.global_pool
        self.graph_encoder_choice = args.graph_encoder_choice
        self.pkgencoder = PKGEncoder(gencoder_dim_list=self.gencoder_dim_list, gencoder_mess_dropout=self.gencoder_mess_dropout,
                                     n_layers_gencoder=self.n_layers_gencoder, edge_dim = self.relation_dim,
                                     global_pool=self.global_pool, graph_encoder_choice = self.graph_encoder_choice)

        self.temperature_init_value = args.temperature_init_value
        self.cooccur_gnn_layers = args.cooccur_gnn_layers

        self.patient_weight_layer = nn.Linear(self.patient_dim + 2 * np.sum(np.array(self.gencoder_dim_list)), 2)
        self.cosine_classifier = CosineClassifier(self.patient_dim + np.sum(np.array(self.gencoder_dim_list)),
                                                  args.label_num, self.temperature_init_value,
                                                  self.cooccur_gnn_layers)


        self.relation_embed = nn.Embedding(self.relation_num+1, self.relation_dim,padding_idx=0)
        self.node_embed = nn.Embedding(self.node_num+1, self.entity_dim,padding_idx=0)
        self.W_R = nn.Parameter(torch.Tensor(self.relation_num + 1, self.entity_dim, self.relation_dim))
        self.ehrcode_embed = nn.Embedding(self.code_num, args.code_dim,padding_idx=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.train_dropout_rate = args.train_dropout_rate
        self.dropout = nn.Dropout(self.train_dropout_rate)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.patient_weight_layer.weight)
        nn.init.zeros_(self.patient_weight_layer.bias)
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.node_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.ehrcode_embed.weight)

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs,
                                                                        x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss

    def calc_gcl_loss(self, x_batch, s_batch, s_batch_dim2, pkg_batch, edgebindex_batch, batch_mask):
        sequence_embedding = torch.matmul(x_batch, self.ehrcode_embed.weight)

        seq_attention, sequence_embedding_final = self.gru(sequence_embedding, s_batch, batch_mask, self.device)
        p_node, keep_sample_node, view_node = self.sampler_node(pkg_batch, self.node_embed, sequence_embedding_final)
        p_edge, keep_sample_edge, view_edge = self.sampler_edge(pkg_batch, self.node_embed, sequence_embedding_final, edgebindex_batch)
        output_node = self.pkgencoder.forward_gcl(view_node,  self.W_R, self.relation_embed, mode ="pre-training")
        output_edge = self.pkgencoder.forward_gcl(view_edge,  self.W_R, self.relation_embed, mode ="pre-training")

        cl_loss = self.loss_cl(output_node, output_edge)

        return cl_loss, p_node, keep_sample_node, p_edge, keep_sample_edge, sequence_embedding_final, seq_attention

    def calc_logit(self,x_batch, s_batch, s_batch_dim2, pkg_batch, edgebindex_batch, cooccur_matrix, batch_mask):
        sequence_embedding = torch.matmul(x_batch, self.ehrcode_embed.weight)
        seq_attention, sequence_embedding_final = self.gru(sequence_embedding, s_batch, batch_mask, self.device)
        p_node, keep_sample_node, view_node = self.sampler_node(pkg_batch, self.node_embed, sequence_embedding_final)
        p_edge, keep_sample_edge, view_edge = self.sampler_edge(pkg_batch, self.node_embed, sequence_embedding_final, edgebindex_batch)
        output_node = self.pkgencoder.forward_logits(view_node,  self.W_R, self.relation_embed, mode ="train")
        output_edge = self.pkgencoder.forward_logits(view_edge,  self.W_R, self.relation_embed, mode ="train")

        all_embedding = torch.cat((sequence_embedding_final, output_node, output_edge), dim=1)
        view_attention = torch.softmax(self.patient_weight_layer(all_embedding),1)
        graph_embedding = view_attention[:,0].view(-1,1) * output_node + view_attention[:,1].view(-1,1) * output_edge
        patient_embedding_final = torch.cat((sequence_embedding_final, graph_embedding), dim=1)
        patient_embedding_final = self.dropout(patient_embedding_final)
        logits = self.cosine_classifier(input= patient_embedding_final, cooccur_matrix =cooccur_matrix)

        return logits, p_node, keep_sample_node, p_edge, keep_sample_edge, seq_attention


    def forward(self, mode, *input):
        if mode == 'calc_gcl_loss':
            return self.calc_gcl_loss(*input)
        elif mode == 'calc_logit':
            return self.calc_logit(*input)





