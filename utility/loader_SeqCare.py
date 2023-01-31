import os
import random
import collections
import pickle

import torch
import numpy as np
import pandas as pd
import math
from torch_geometric.loader import DataLoader

class DataLoaderSeqCare(object):
    def __init__(self, args, logging):
        self.model_input_data_dir = args.model_input_data_dir

        self.sequence_batch_size = args.sequence_batch_size
        self.pretrain_batch_size = args.pretrain_batch_size

        self.train_sequence_file_path = self.model_input_data_dir + "train.pkl"
        self.val_sequence_file_path = self.model_input_data_dir + "val.pkl"
        self.test_sequence_file_path = self.model_input_data_dir + "test.pkl"
        self.pkg_file_path = self.model_input_data_dir + "pkg_train_val_test_dict.pkl"
        self.edgenum_file_path = self.model_input_data_dir + "edgenum_train_val_test_dict.pkl"
        self.cooccur_matrix_file_path = self.model_input_data_dir + "label_occurance_graph.pkl"

        self.concept_file_path = args.concept_file_path
        self.relation_file_path = args.relation_file_path
        self.triple_file_path = args.triple_file_path
        self.triple_corrupt_file_path = args.triple_corrupt_file_path

        self.cuda_choice = args.cuda_choice
        self._load_input_data()
        self._load_kg_data()

    def _load_input_data(self):
        with open (self.train_sequence_file_path,"rb") as fin:
            data_dict_train = pickle.load(fin)
        with open (self.val_sequence_file_path,"rb") as fin:
            data_dict_val = pickle.load(fin)
        with open (self.test_sequence_file_path,"rb") as fin:
            data_dict_test = pickle.load(fin)
        with open (self.pkg_file_path,"rb") as fin:
            self.pkg_train_val_test_dict = pickle.load(fin)
        with open (self.edgenum_file_path,"rb") as fin:
            self.edgenum_train_val_test_dict = pickle.load(fin)
        with open (self.cooccur_matrix_file_path,"rb") as fin:
            self.cooccur_matrix = pickle.load(fin)

        self.max_visit_len = data_dict_train["sequence_all_array"][0].get_shape()[0]
        self.code_num = data_dict_train["sequence_all_array"][0].get_shape()[1]


        self.sequence_train_array = [data_dict_train["sequence_all_array"],data_dict_train["sequence_len_array"],
                                     data_dict_train["sequence_len_dim2_array"],data_dict_train["label_array"],data_dict_train["mask"],data_dict_train["seq_time"]]
        self.sequence_val_array = [data_dict_val["sequence_all_array"], data_dict_val["sequence_len_array"],
                                     data_dict_val["sequence_len_dim2_array"], data_dict_val["label_array"],data_dict_val["mask"],data_dict_val["seq_time"]]
        self.sequence_test_array = [data_dict_test["sequence_all_array"], data_dict_test["sequence_len_array"],
                                     data_dict_test["sequence_len_dim2_array"], data_dict_test["label_array"],data_dict_test["mask"],data_dict_test["seq_time"]]

        self.sequence_array_dict ={}
        self.sequence_array_dict["train"] = self.sequence_train_array
        self.sequence_array_dict["val"] =self.sequence_val_array
        self.sequence_array_dict["test"] = self.sequence_test_array


    def _load_kg_data(self):
        self.concept_dic = {}
        with open(self.concept_file_path, "r") as f:
            concept_list = f.readlines()
            concept_list = [line.strip() for line in concept_list]

            for line in concept_list[1:]:
                line_list = line.split('\t')
                renmen_node_name = line_list[0]
                item_id = int(line_list[1])
                self.concept_dic[renmen_node_name] = item_id
        self.kg_node_num = len(list(self.concept_dic.keys()))

        self.relation_dic = {}
        with open(self.relation_file_path, "r") as f:
            relation_list = f.readlines()
            relation_list = [line.strip() for line in relation_list]

            for line in relation_list[1:]:
                line_list = line.split('\t')
                renmen_rel_name = line_list[0]
                item_id = int(line_list[1])
                self.relation_dic[renmen_rel_name] = item_id
        self.kg_edge_num = len(list(self.relation_dic.keys()))

        self.all_h_list, self.all_t_list, self.all_r_list, self.all_v_list = [], [], [], []
        with open(self.triple_file_path, "r") as f:
            triple_list = f.readlines()
            triple_list = [line.strip() for line in triple_list]

            for line in triple_list[1:]:
                sp_list = line.split(' ')
                h_id = int(sp_list[0])
                t_id = int(sp_list[1])
                r_id = int(sp_list[2])
                self.all_h_list.append(h_id)
                self.all_t_list.append(t_id)
                self.all_r_list.append(r_id)
                self.all_v_list.append(1.0)


        self.all_corrupt_list = []
        with open(self.triple_corrupt_file_path, "r") as f:
            triple_corrupt_list = f.readlines()
            triple_corrupt_list = [line.strip() for line in triple_corrupt_list]
            for line in triple_corrupt_list[1:]:
                sp_list = line.split(' ')
                h_id = int(sp_list[0])
                corrt_id = int(sp_list[1])
                r_id = int(sp_list[2])
                self.all_corrupt_list.append(corrt_id)

    def generate_edge_batch_index(self,edgenum_batch):
        len_batch = len(edgenum_batch)
        edgebinex_batch = []
        for i in range(len_batch):
            edgebinex_batch += [i] * edgenum_batch[i]
        return edgebinex_batch

    def sequence_batch_iter(self,flag,args,shuffle = False,re_index = None):
        sequence_all_array = self.sequence_array_dict[flag]
        pkg_list = self.pkg_train_val_test_dict[flag]
        edgenum_list = self.edgenum_train_val_test_dict[flag]
        patient_num  = len(sequence_all_array[0])
        batch_num = math.ceil(patient_num / self.sequence_batch_size)

        index_array = list(range(patient_num))

        if shuffle:
            np.random.shuffle(index_array)
        pkg_loader = DataLoader(pkg_list, batch_size=self.sequence_batch_size)
        pkg_iter = iter(pkg_loader)

        for i in range(batch_num):
            indices = index_array[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]

            x_batch_origin = sequence_all_array[0][i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            x_batch = np.array([sparse_matrix.toarray() for sparse_matrix in x_batch_origin])
            s_batch = sequence_all_array[1][indices]
            s_batch_dim2 = sequence_all_array[2][indices]
            y_batch = sequence_all_array[3][indices]
            batch_mask = sequence_all_array[4][indices]
            seq_time_batch = sequence_all_array[5][indices]
            edgenum_batch = edgenum_list[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            edgebindex_batch = self.generate_edge_batch_index(edgenum_batch)


            pkg_batch = next(pkg_iter)

            yield x_batch, s_batch, s_batch_dim2, y_batch, seq_time_batch, pkg_batch, edgebindex_batch, batch_mask















