import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import json
import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from model.SeqCare import SeqCare
from utility.parser_SeqCare import *
from utility.log_helper import *
from utility.loader_SeqCare import DataLoaderSeqCare
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,auc, roc_curve, precision_recall_curve
from scipy import interp
from utility.criterion import DistillKL
import math

def get_metrics(predict_all,targets_all, flag, epoch_id, result_path):
    result = dict()
    result["pr"] = get_metrics_pr(predict_all, targets_all)
    result["auc"] = get_metric_auc(predict_all, targets_all)

    with open(result_path, "a") as f:
        if flag == "train":
            f.write("train_epoch{}:".format(epoch_id) + str(result) + '\n')
        elif flag == "val":
            f.write("val_epoch{}:".format(epoch_id) + str(result) + '\n')
        elif flag == "test":
            f.write("test_epoch{}:".format(epoch_id)+str(result)+'\n')
    
    return result["pr"],result["auc"]

def get_metrics_pr(predict_all, targets_all):
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(predict_all.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            targets_all[:, i], predict_all[:, i])
        precision[i][np.isnan(precision[i])] = 0
        recall[i][np.isnan(recall[i])] = 0
        pr_auc[i] = auc(recall[i], precision[i])
        
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        targets_all.ravel(), predict_all.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    macro = 0.0
    for i in range(predict_all.shape[1]):
        macro += pr_auc[i]
    pr_auc['macro'] = macro / predict_all.shape[1]

    return pr_auc

def get_metric_auc(predict_all,targets_all):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(predict_all.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(
            targets_all[:, i], predict_all[:, i])
        fpr[i][np.isnan(fpr[i])] = 0
        tpr[i][np.isnan(tpr[i])] = 0
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(
        targets_all.ravel(), predict_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate(
        [fpr[i] for i in range(predict_all.shape[1])]))  
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(predict_all.shape[1]):  
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= predict_all.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc

def evaluate(args,device,model,dataLoader,result_path, cooccur_matrix, flag,epoch):
    model.eval()

    y_true_evaluate = []
    y_pred_evaluate = []
    with torch.no_grad():
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch, seq_time_batch,
        pkg_batch, edgebindex_batch, batch_mask) in enumerate(dataLoader.sequence_batch_iter(flag=flag,args=args)):

            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)


            pkg_batch = pkg_batch.to(device)
            edgebindex_batch = torch.LongTensor(edgebindex_batch).to(device)
            batch_mask = torch.FloatTensor(batch_mask).to(device)

            logits, p_node, keep_sample_node, p_edge, keep_sample_edge, seq_attention = model('calc_logit', x_batch, s_batch, s_batch_dim2, pkg_batch, edgebindex_batch, cooccur_matrix, batch_mask)
            real_logits = torch.sigmoid(logits)

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_evaluate.append(labels_cpu)
            y_pred_evaluate.append(logits_cpu)

    y_true_evaluate = np.vstack(y_true_evaluate)
    y_pred_evaluate = np.vstack(y_pred_evaluate)
    prauc_dict,auc_dict = get_metrics(y_pred_evaluate, y_true_evaluate, flag, epoch, result_path)

    return prauc_dict["micro"],prauc_dict["macro"],auc_dict["micro"],auc_dict["macro"]

def union(args):
    gamma = args.gamma
    alpha = 1.0-gamma
    temp_KL = args.temp_KL

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    dataLoader = DataLoaderSeqCare(args, logging)
    model_t = SeqCare(args, dataLoader.kg_node_num, dataLoader.code_num, dataLoader.kg_edge_num,
                   dataLoader.max_visit_len)


    model_t.to(device)
    logging.info(model_t)
    with open(args.save_dir+"params.json",mode = "w") as f:
        json.dump(args.__dict__,f,indent=4)

    grouped_parameters_t = [
        {'params': [p for n, p in model_t.named_parameters() if ('sampler_node' in n)],
          'lr': args.sampler_node_lr},
        {'params': [p for n, p in model_t.named_parameters() if ('sampler_edge' in n)],
          'lr': args.sampler_edge_lr},
        {'params': [p for n, p in model_t.named_parameters() if (not (('sampler_edge' in n) or ('sampler_node' in n)))],
         'lr': args.base_lr}
    ]

    optimizer_cl = optim.Adam(grouped_parameters_t)
    optimizer_seq = optim.Adam(grouped_parameters_t)
    model_path =os.path.join(args.save_dir, 'model.pt')
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    result_path = args.save_dir + "metrics.txt"
    best_dev_epoch = 0
    best_dev_auc, final_micro_prauc, final_macro_prauc, final_micro_auroc, final_macro_auroc = 0.0, 0.0, 0.0, 0.0, 0.0

    cooccur_matrix = dataLoader.cooccur_matrix
    cooccur_matrix = cooccur_matrix.to(device)

    for epoch in range(1, args.n_epoch_pretrain + 1):
        time0 = time()
        model_t.train()
        gcl_total_loss = 0.0
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch, seq_time_batch,
        pkg_batch, edgebindex_batch, batch_mask) in enumerate(dataLoader.sequence_pretrain_iter(flag="train", args=args)):
            optimizer_cl.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            pkg_batch = pkg_batch.to(device)
            edgebindex_batch = torch.LongTensor(edgebindex_batch).to(device)
            batch_mask = torch.FloatTensor(batch_mask).to(device)
            cl_loss, p_node, keep_sample_node, p_edge, keep_sample_edge, sequence_embedding_final, seq_attention = model_t('calc_gcl_loss',x_batch, s_batch,
                                                                                s_batch_dim2, pkg_batch, edgebindex_batch, batch_mask)
            cl_loss.backward()
            gcl_total_loss += cl_loss.item()
            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model_t.parameters(), args.clip)

            optimizer_cl.step()
        logging.info(
                'Sequence Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time0,gcl_total_loss))


    for epoch in range(1,args.n_epoch+1):
        time0 = time()
        model_t.train()
        sequence_total_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch, seq_time_batch,
        pkg_batch, edgebindex_batch, batch_mask) in enumerate(dataLoader.sequence_batch_iter(flag="train", args=args)):
            optimizer_seq.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            pkg_batch = pkg_batch.to(device)
            edgebindex_batch = torch.LongTensor(edgebindex_batch).to(device)
            batch_mask = torch.FloatTensor(batch_mask).to(device)
            logits, p_node, keep_sample_node, p_edge, keep_sample_edge, seq_attention = model_t('calc_logit',x_batch, s_batch, s_batch_dim2, pkg_batch, edgebindex_batch, cooccur_matrix, batch_mask)
            real_logits = torch.sigmoid(logits)
            sequence_loss = loss_func(logits,y_batch.float())

            sequence_loss.backward()
            sequence_total_loss += sequence_loss.item()

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model_t.parameters(), args.clip)
            optimizer_seq.step()

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_train.append(labels_cpu)
            y_pred_train.append(logits_cpu)

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)
        logging.info(
                'Sequence Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time0,sequence_total_loss))

    model_s = SeqCare(args, dataLoader.kg_node_num, dataLoader.code_num, dataLoader.kg_edge_num,
                   dataLoader.max_visit_len)
    model_s.to(device)
    grouped_parameters_s = [
        {'params': [p for n, p in model_s.named_parameters() if ('sampler_node' in n)],
          'lr': args.sampler_node_lr},
        {'params': [p for n, p in model_s.named_parameters() if ('sampler_edge' in n)],
          'lr': args.sampler_edge_lr},
        {'params': [p for n, p in model_s.named_parameters() if (not (('sampler_edge' in n) or ('sampler_node' in n)))],
         'lr': args.base_lr}
    ]
    optimizer_seq_s = optim.Adam(grouped_parameters_s)
    model_t.eval()
    criterion_div = DistillKL(temp_KL)
    for p in model_t.parameters():
        p.requires_grad = False


    for epoch in range(1, args.n_epoch_kl + 1):
        time0 = time()
        model_s.train()
        final_total_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch, seq_time_batch,
        pkg_batch, edgebindex_batch, batch_mask) in enumerate(dataLoader.sequence_batch_iter(flag="train", args=args)):
            optimizer_seq_s.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            pkg_batch = pkg_batch.to(device)
            edgebindex_batch = torch.LongTensor(edgebindex_batch).to(device)
            batch_mask = torch.FloatTensor(batch_mask).to(device)
            logits_s, p_node_s, keep_sample_node_s, p_edge_s, keep_sample_edge_s, seq_attention_s = model_s('calc_logit',x_batch, s_batch, s_batch_dim2, pkg_batch, edgebindex_batch, cooccur_matrix, batch_mask)
            logits_t, p_node_t, keep_sample_node_t, p_edge_t, keep_sample_edge_t, seq_attention_t = model_t('calc_logit', x_batch, s_batch, s_batch_dim2, pkg_batch,
                                                edgebindex_batch, cooccur_matrix, batch_mask)
            real_logits = torch.sigmoid(logits_s)
            sequence_loss = loss_func(logits_s,y_batch.float())
            div_loss = criterion_div(logits_s, logits_t)
            final_loss = gamma * sequence_loss + alpha * div_loss

            final_loss.backward()
            final_total_loss += final_loss.item()

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), args.clip)
            optimizer_seq_s.step()

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_train.append(labels_cpu)
            y_pred_train.append(logits_cpu)

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)
        micro_auc, macro_auc = get_metrics(y_pred_train, y_true_train, "train", epoch, result_path)

        logging.info(
                'Sequence Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time0,sequence_total_loss))

        time1 = time()
        dev_micro_prauc,dev_macro_prauc,dev_micro_auroc,dev_macro_auroc = evaluate(args, device, model_s, dataLoader, result_path, cooccur_matrix, "val",
                                                 epoch)
        logging.info(
            'Val Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))

        time1 = time()
        test_micro_prauc,test_macro_prauc,test_micro_auroc,test_macro_auroc = evaluate(args, device, model_s, dataLoader, result_path, cooccur_matrix, "test",
                                                   epoch)
        logging.info(
            'Test Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))
        if dev_macro_prauc >= best_dev_auc:
            best_dev_auc = dev_macro_prauc
            best_dev_epoch = epoch
            final_micro_prauc = test_micro_prauc
            final_micro_auroc = test_micro_auroc
            final_macro_auroc = test_macro_auroc
            final_macro_prauc = test_macro_prauc
            torch.save(model_t.state_dict(), model_path)
            print(f'model saved to {model_path}')

        logging.info("Epoch: {}".format(epoch))
        logging.info('best test micro prauc: {:.4f}'.format(final_micro_prauc))
        logging.info('best test macro prauc: {:.4f}'.format(final_macro_prauc))
        logging.info('best test micro auroc: {:.4f}'.format(final_micro_auroc))
        logging.info('best test macro auroc: {:.4f}'.format(final_macro_auroc))
        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('best test auc: {:.4f} (at epoch {})'.format(final_macro_prauc, best_dev_epoch))
    logging.info('final test micro prauc: {:.4f}'.format(final_micro_prauc))
    logging.info('final test macro prauc: {:.4f}'.format(final_macro_prauc))
    logging.info('final test micro auroc: {:.4f}'.format(final_micro_auroc))
    logging.info('final test macro auroc: {:.4f}'.format(final_macro_auroc))
    logging.info('{:.4f},{:.4f},{:.4f},{:.4f}'.format(final_micro_prauc, final_macro_prauc, final_micro_auroc,
                                                      final_macro_auroc))

if __name__ == "__main__":
    args = parse_seqcare_args()
    union(args)


