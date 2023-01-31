import argparse
import datetime

def parse_seqcare_args():
    parser = argparse.ArgumentParser(description="Run SeqCare.")

    parser.add_argument('--seed', type=int, default=629,
                        help='Random seed.')

    parser.add_argument('--model_input_data_dir', nargs='?', default='./data_preprocess/model_input_data_new_629/',
                        help='Input model input data path')
    parser.add_argument('--kg_data_dir', nargs='?', default='./data_preprocess/graph_preprocessed/',
                        help='Input kg data path.')

    parser.add_argument('--concept_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/entity2id_1hop.txt',
                        help='Path of concept file.')
    parser.add_argument('--relation_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/relation2id_1hop.txt',
                        help='Path of relation file.')
    parser.add_argument('--triple_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/train2id_1hop.txt',
                        help='Path of triple file.')
    parser.add_argument('--triple_corrupt_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/train2id_1hop_corrupt.txt',
                        help='Path of corrupt triple file.')

    parser.add_argument('--cuda_choice', nargs='?', default='cuda:0',
                        help='GPU choice.')

    parser.add_argument('--sequence_batch_size', type=int, default=64,
                        help='sequence batch size.')
    parser.add_argument('--pretrain_batch_size', type=int, default=64,
                        help='pretrain batch size.')

    parser.add_argument('--entity_dim', type=int, default=64,
                        help='Node Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=96,
                        help='Relation Embedding size.')


    parser.add_argument('--sampler_gnn_layers', type=int, default=1,
                        help='Sampler GNN Layer Num')
    parser.add_argument('--sampler_node_lr', type=float, default=2e-5,
                        help='Node Sampler Learning rate.')
    parser.add_argument('--sampler_edge_lr', type=float, default=2e-5,
                        help='Edge Sampler Learning rate.')


    parser.add_argument('--gencoder_dim_list', nargs='?', default='[64]',
                        help='Output sizes of every aggregation layer in Graph Encoder.')
    parser.add_argument('--gencoder_mess_dropout', type=float, default=0.5,
                        help='Dropout probability for Graph Encoder w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')

    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base Learning rate.')

    parser.add_argument('--n_epoch_pretrain', type=int, default=200,
                        help='Number of pretraining epoch.')
    parser.add_argument('--label_num', type=int, default=150,
                        help='Number of multi label.')

    parser.add_argument('--gru_dropout_prob', type=float, default=0.,
                        help='GRU Dropout rate')
    parser.add_argument('--gru_layer', type=int, default=1,
                        help='GRU Layer Num')
    parser.add_argument('--gru_hidden_size', type=int, default=128,
                        help='GRU hidden size')
    parser.add_argument('--code_dim', type=int, default=64,
                        help='EHR Code dimension')

    parser.add_argument('--train_dropout_rate', type=float, default=0.5,
                        help='Train Dropout rate')


    parser.add_argument('--temperature_init_value',type=int, default=100,
                        help='Temperature Init Value')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing sequence loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating multi-label.')

    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=30, type=int,
                        help='stop training if dev does not increase for N epochs')


    parser.add_argument('--global_pool', nargs='?', default='mean', choices=['mean'],
                        help='Graph global pooling method')
    parser.add_argument('--graph_encoder_choice', nargs='?', default='CAGAT', choices=['CAGAT'],
                        help='Graph encoder choice')
    parser.add_argument('--cooccur_gnn_layers', type=int, default=1,
                        help='Cooccur GNN Layer Num')


    parser.add_argument('--n_epoch_kl', type=int, default=100,
                        help='Number of sequential self-distillation.')
    parser.add_argument('--temp_KL', type=int, default=4,
                        help='Temperature Init Value for self-distillation.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Weight for Spervised loss during self-distillation')
    parser.add_argument('--n_epoch', type=int, default=30,
                        help='Number of epoch.')

    args = parser.parse_args()

    save_dir = 'trained_model/SeqCare/{}/edim{}_rdim{}_sequenceBatch{}_gDim{}_samnodelr{}_samedgelr{}_baselr{}_gruDim{}_codeDim{}_preepoch{}_cchoice-{}_preBatch{}_ablation-{}_kmask{}_klepoch{}_gamma{}_kltemp{}_trepoch{}_seed{}_{}/'.format(
        str(datetime.datetime.now().strftime('%Y-%m-%d')), args.entity_dim, args.relation_dim,
        args.sequence_batch_size,
        args.gencoder_dim_list,
        args.sampler_node_lr,args.sampler_edge_lr ,args.base_lr,
        args.gru_hidden_size,
        args.code_dim,
        args.n_epoch_pretrain,
        args.classifier_choice,
        args.pretrain_batch_size,
        "-".join(args.ablation),
        args.know_mask_rate,
        args.n_epoch_kl,
        args.gamma,
        args.temp_KL,
        args.n_epoch,
        args.seed,args.cuda_choice)
    args.save_dir = save_dir

    return args


