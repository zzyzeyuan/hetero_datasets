import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from model import HINormer
from utils.data import load_data
from utils.pytorchtools import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append('utils/')

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def fun_dblp(args, dl, adjM, node_cnt):
    device = args.device
    nodes_num = sum(dl.nodes['count'].values())
    edge_index = torch.zeros(2,1, dtype=int).to(device)
    edge_type = torch.zeros(1,1, dtype=int).squeeze(0).to(device)
    
    for i in range(nodes_num):
        
        node_indices = adjM[i].indices.tolist()
        
        sub_edge_index = torch.zeros(2, len(node_indices), dtype=int).to(device)
        sub_edge_index[0] = i
        sub_edge_index[1] = torch.tensor(node_indices, dtype=int).to(device)
        edge_index = torch.cat((edge_index, sub_edge_index), dim=1)

        sub_edge_type = torch.zeros(1, len(node_indices), dtype=int).squeeze(0).to(device)
        for j in range(len(node_indices)):
            """
            # A-P, P-A: 0
            # P-T, T-P: 1
            # C-P, P-C: 2
            # 起点是author 终点是paper ==> edge type 是0    
            # 只要开头是author 就认为是  A-P
            """
            if i < sum(node_cnt[:1]): # src is author, edge_type always 0
                sub_edge_type = torch.zeros(1, len(node_indices)).squeeze(0).to(device)
                break
            
            elif i >= sum(node_cnt[:1]) and i < sum(node_cnt[:2]): # src is paper
                if node_indices[j] < sum(node_cnt[:1]): # dst is author
                    sub_edge_type[j] = 0
                elif node_indices[j] >= sum(node_cnt[:2]) and node_indices[j] < sum(node_cnt[:3]): # dst is term
                    sub_edge_type[j] = 1
                elif node_indices[j] >= sum(node_cnt[:3]) and node_indices[j] < sum(node_cnt): # dst is conference
                    sub_edge_type[j] = 2

            elif i >= sum(node_cnt[:2]) and i < sum(node_cnt[:3]): # src is term
                # sub_edge_type[j] = 1
                sub_edge_type = torch.ones(1, len(node_indices)).squeeze(0).to(device)
                break
            
            elif i >= sum(node_cnt[:3]) and i < sum(node_cnt): # src is conference
                # sub_edge_type[j] = 2
                sub_edge_type = torch.full((1, len(node_indices)), 2).squeeze(0).to(device)
                break
        edge_type = torch.cat((edge_type, sub_edge_type))


    
    print(edge_index)
    print('edge idx shape: ', edge_index.shape)
    print('=='*20)
    print(edge_type)
    print('edge_type.shape: ', edge_type.shape)
    # exit()
    torch.save((edge_index[:, 1:], edge_type[1:]), 'pt/dblp.pt')


def get_sparse_format_data(dataset_name):
    file = 'data/' + str(dataset_name) + '/link.dat'
    print(file)
    sparse_data = {'src': [], 'dst': [], 'type': [], 'weight': []}
    with open(file, 'r') as f:
        for line in f:
            src, dst, link_type, weight = line.strip().split('\t')
            sparse_data['src'].append(int(src))
            sparse_data['dst'].append(int(dst))
            sparse_data['type'].append(int(link_type))
            sparse_data['weight'].append(float(weight))
    
    return sparse_data['type']

def run_model_DBLP(args):

    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')
    print('Dataset Name: ', args.dataset)
    print('args: ', args)
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)

    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    
    node_cnt = [features.shape[0] for features in features_list]
    # etype = get_sparse_format_data(args.dataset)
    # etype = torch.tensor(etype).to(device)
    sum_node = 0
    for x in node_cnt:
        sum_node += x

    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    g = dgl.from_scipy(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    t1 = time.time()
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    
    t2 = time.time()
    print('Got e_feat, time consume: %fs'%(t2 - t1))
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    
    """
    ======>DBLP 
    g.edges():
    (
        tensor([    0,     0,     1,  ..., 26125, 26126, 26127]), 
        tensor([ 6421, 10514,  6422,  ..., 26125, 26126, 26127])
        )
    DBLP e_feat:
    tensor([0, 0, 0,  ..., 6, 6, 6], device='cuda:7')        e_feat shape:  torch.Size([265694])
    ============================================================================================
    ======>Freebase 
    g.edges():
     (
        tensor([     0,      0,      0,  ..., 180095, 180096, 180097]), 
        tensor([  8665,  38654, 143175,  ..., 180095, 180096, 180097])
        )
    Freebase e_feat:
    tensor([ 0,  0,  2,  ..., 36, 36, 36], device='cuda:2')  e_feat shape:  torch.Size([1623206])
    ============================================================================================
    ======>ACM
    g edges():
    (
        tensor([    0,     0,     0,  ..., 10939, 10940, 10941]), 
        tensor([  179,   526,   583,  ..., 10939, 10940, 10941])
        )
    ACM e_feat:
    tensor([0, 1, 1,  ..., 8, 8, 8], device='cuda:2') torch.Size([558748])
    """
   
    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]
    
    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)
    num_classes = dl.labels_train['num_classes']
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)

    for i in range(args.repeat):
        
        net = HINormer(g, 
                       args.dataset,
                       num_classes, 
                       in_dims, 
                       args.hidden_dim, 
                       args.ffn_dim, 
                       args.cut, 
                       args.num_layers, 
                       args.num_gnns, 
                       args.num_heads, 
                       args.dropout, 
                       args.attn_dropout,
                       temper=args.temperature, 
                       num_type=len(node_cnt), 
                       device=args.device)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, save_path='checkpoint/HINormer_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            logits = net(features_list, train_idx, type_emb, node_type, e_feat, args.l2norm)
            logp = F.log_softmax(logits, 1)
            # logp = logits
            # print('logp shape: ', logp.shape)
            # print(labels[train_idx].shape)
            # exit()
            train_loss = F.nll_loss(logp, labels[train_idx])

            # autograd
            optimizer.zero_grad() 
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()

            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, val_idx, type_emb, node_type, e_feat, args.l2norm)
                logp = F.log_softmax(logits, 1)
                # logp = logits
                val_loss = F.nll_loss(logp, labels[val_idx])
                pred = logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))
    
            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                # print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/HINormer_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device)))
        net.eval()
        with torch.no_grad():
            logits = net(features_list, test_idx, type_emb, node_type, e_feat, args.l2norm)
            test_logits = logits
            # print(test_logits.shape)
            # test_logits[1417][0] = test_logits[1417][0] + 3.0
            # test_logits[1418][0] = test_logits[1418][0] + 3.0
            # print('test logits: ', test_logits)
            # exit()
            if args.mode == 1:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                result = dl.evaluate_valid(pred, dl.labels_test['data'][test_idx])
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']

    print('args: ', args)
    print('Micro-f1: ', micro_f1)
    print('Macro-f1: ', macro_f1)
    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))
    

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='HINormer')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--ffn-dim', type=int, default=64)
    ap.add_argument('--hidden-dim', type=int, default=512,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--dataset', type=str, default = 'DBLP', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
    ap.add_argument('--num-gnns', type=int, default=4, help='The number of layers of both structural and heterogeneous encoder')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--cut', type=int, default=4, help='cut_dim = hidden_dim // cut')
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--attn-dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    # ap.add_argument('--len-seq', type=int, default=200, help='The length of node sequence.')
    ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
    ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
    ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
    # ap.add_argument('--beta', type=float, default=1.0, help='Weight of heterogeneity-level attention score')

    
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    random.seed(args.seed)
    run_model_DBLP(args)

