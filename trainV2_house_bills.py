import argparse
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
import random
import itertools
import pickle
from torch_geometric.data import Data
from torch_geometric.logging import log
from torch_geometric.data import Data
from scipy.stats import pearsonr
from conformal.helper import HGNN
from conformal.utils import run_conformal_classification
from conformal.helper import ConfHNN, ConfMLP,HyperedgeDegreePredictor,edge_task
from conformal.utils import InfoNCE
from torch_scatter import scatter_mean
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='house-bills', choices = ['house-bills', 'house-committees', 'walmart-trips','cora','dblp', 'citeseer', 'pubmed','congress-bills']) #'house-bills'
parser.add_argument('--types', type=str, default=None, choices = [None, 'CA', 'CF'])
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--base_dropout', type=float, default=0.1)
parser.add_argument('--synthetic_feature_dim', type=int, default=128)
parser.add_argument('--model', type=str, default='hgnn', choices = ['hgnn',])
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1500)#5000
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--conftr_calib_holdout', action='store_true', default = False)

parser.add_argument('--conf_correct_model', type=str, default='hnn', choices = ['hnn', 'mlp'])
parser.add_argument('--calibrator', type=str, default='NULL', choices = ['TS', 'VS', 'ETS', 'CaGCN', 'GATS'])

parser.add_argument('--quantile', action='store_true', default = False)
parser.add_argument('--bnn', action='store_true', default = False)

parser.add_argument('--target_size', type=int, default=0)
parser.add_argument('--confnn_hidden_dim', type=int, default=128)
parser.add_argument('--confgnn_num_layers', type=int, default=3)
parser.add_argument('--confgnn_base_model', type=str, default='HGNN', choices = ['HGNN'])
parser.add_argument('--confgnn_lr', type=float, default=1e-3)
parser.add_argument('--confgnn_weight_decay', type=float, default=5e-4)
parser.add_argument('--confgnn_dropout', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--size_loss_weight', type=float, default=4)
parser.add_argument('--reg_loss_weight', type=float, default=1)

parser.add_argument('--not_save_res', action='store_true', default = False)
parser.add_argument('--num_runs', type=int, default=20)
parser.add_argument('--retrain', action='store_true', default = False)
parser.add_argument('--verbose', action='store_true', default = False)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--cond_cov_loss', action='store_true', default = False)
parser.add_argument('--conftr', action='store_true', default = True)
parser.add_argument('--temperature', type=float, default=0.3)#0.2
parser.add_argument('--aug_ratio', type=float, default=0.3)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--train_fraction', type=float, default=0.2)
parser.add_argument('--valid_fraction', type=float, default=0.1)
parser.add_argument('--max_calib_size', type=int, default=1000)
parser.add_argument('--valid_split_seed', type=int, default=0)
parser.add_argument('--conformal_trials', type=int, default=100)
parser.add_argument('--edge_topk', type=int, default=10000)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--coverage_warmup_epochs', type=int, default=300)
parser.add_argument('--contrastive_loss_weight', type=float, default=0.1)
parser.add_argument('--late_contrastive_loss_weight', type=float, default=0.01)
parser.add_argument('--non_conftr_size_loss_weight', type=float, default=1.0)
parser.add_argument('--raps_lam_reg', type=float, default=0.01)


parser.add_argument('--calib_fraction', type=float, default=0.5)
parser.add_argument('--optimize_conformal_score', type=str, default='aps', choices = ['aps', 'raps'])
parser.add_argument('--raps_k', type=int, default=1)  # k for RAPS (top-k unpenalized); must be < num_classes


args = parser.parse_args()

global task

task = 'classification'


device = torch.device(args.device)
def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def train(epoch, model, data, optimizer, alpha):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test(model, data, alpha, tau, target_size, size_loss = False):
    model.eval()
    if size_loss:
        pred_raw, ori_pred_raw = model(data.x, data.edge_index,argu='NoCon')
    else:
        pred_raw = model(data)

    pred = pred_raw.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.valid_mask, data.calib_test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    if size_loss:
        out_softmax = F.softmax(pred_raw, dim = 1)
        query_idx = np.where(data.valid_mask.detach().cpu().numpy())[0]
        np.random.seed(args.valid_split_seed)
        np.random.shuffle(query_idx)

        train_train_idx = query_idx[:int(len(query_idx)/2)]
        train_calib_idx = query_idx[int(len(query_idx)/2):]

        n_temp = len(train_calib_idx)
        q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp

        tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
        qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')
        c = torch.sigmoid((out_softmax[train_train_idx] - qhat)/tau)
        size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))

        return accs, pred_raw, size_loss.item()

    return accs, pred_raw

def add_self_loops(hypergraph):
        # Flatten the list of hyperedges to get all node IDs
        all_nodes = set(node for edge in hypergraph for node in edge)
        # Add self-loop for each node
        self_loops = [[node] for node in all_nodes]
        return hypergraph + self_loops
def hypergraph_to_edge_index(hypergraph):
        edge_list = []
        for edge in hypergraph:
            if len(edge) >= 2:
                # Create all possible pairs (i,j) where i ≠ j
                edge_list.extend(itertools.combinations(edge, 2))
        # Make edges undirected: for (i,j), also add (j,i)
        edge_list += [(j, i) for i, j in edge_list]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
def dataset_preprocess(dataset_name,types=None):
    if dataset_name in ['cora','dblp'] and types=='CA': # 'Cora_CC', 'CiteSeer_CC', 'PubMed_CC'
        path = 'data/coauthorship/'+dataset_name+'/'
        path1=path+'hypergraph.pickle'
        with open(path1, 'rb') as handle:
            hypergraph = pickle.load(handle)
        hypergraph = {str(k): v for k, v in hypergraph.items()}
        path1=path+'features.pickle'
        with open(path1, 'rb') as handle:
            features = pickle.load(handle).todense()
            # covert features to pytorch tensor
            features = torch.tensor(features, dtype=torch.float)
        path1=path+'labels.pickle'
        with open(path1, 'rb') as handle:
            labels = pickle.load(handle)
        # convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        # Step 2: Create a mapping from original node IDs to new sequential IDs
        all_nodes = sorted(set(n for edge in hypergraph for n in edge))
        node_id_map = {orig_id: new_id for new_id, orig_id in enumerate(all_nodes)}
        num_nodes = len(node_id_map)
        hypergraph=[v for k, v in hypergraph.items()]
    elif dataset_name in ['cora', 'citeseer', 'pubmed'] and types=='CF':
        path = 'data/cocitation/'+dataset_name+'/'
        path1=path+'hypergraph.pickle'
        with open(path1, 'rb') as handle:
            hypergraph = pickle.load(handle)
        # iterate over the hypergraph and convert the keys to str
        hypergraph = {str(k): v for k, v in hypergraph.items()}

        path1=path+'features.pickle'
        with open(path1, 'rb') as handle:
            features = pickle.load(handle).todense()
        # covert features to pytorch tensor
        features = torch.tensor(features, dtype=torch.float)
        path1=path+'labels.pickle'
        with open(path1, 'rb') as handle:
            labels = pickle.load(handle)
        # convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
                # Step 2: Create a mapping from original node IDs to new sequential IDs
        all_nodes = sorted(set(n for edge in hypergraph for n in edge))
        node_id_map = {orig_id: new_id for new_id, orig_id in enumerate(all_nodes)}
        num_nodes = len(node_id_map)
        hypergraph=[v for k, v in hypergraph.items()]
    elif dataset_name in ['house-committees','house-bills','walmart-trips','congress-bills'] and types==None:
        path='data/'+dataset_name+'/'
        path1=path+'hyperedges-'+dataset_name+'.txt'
        hypergraph= []
        with open(path1, 'r') as file:
            for line in file:
                # Split the line by commas, convert each element to an integer, and append it to the list
                hypergraph.append([int(num) for num in line.strip().split(',')])
        # Step 2: Create a mapping from original node IDs to new sequential IDs
        all_nodes = sorted(set(n for edge in hypergraph for n in edge))
        node_id_map = {orig_id: new_id for new_id, orig_id in enumerate(all_nodes)}
        num_nodes = len(node_id_map)

        # Step 3: Re-map node IDs in the hypergraph
        hypergraph = [[node_id_map[n] for n in edge] for edge in hypergraph]


        labels1 = []
        path1=path+'node-labels-'+dataset_name+'.txt'
        with open(path1, 'r') as file:
            for line in file:
                # Split the line by commas, convert each element to an integer, and append it to the list
                labels1.append(int(line.strip()))
        labels=[x-1 for x in labels1]
        features=nn.Embedding(len(labels), args.synthetic_feature_dim).weight.detach().cpu()
        # convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
    # get the lenth of unique labels(torch tensor)
    #import pdb;pdb.set_trace()
    unique_labels = len(torch.unique(labels))
    num_hyperedges= len(hypergraph)
    # convert dict to list
    if dataset_name not in ['house-committees','house-bills','walmart-trips','congress-bills'] and types==None:
        hypergraph = list(hypergraph.values())
    # Step 4: Add self-loops if needed
    hypergraph = hypergraph + [[i] for i in range(num_nodes)]  # or keep your add_self_loops()

    # Step 5: Build the hyperedge_index
    num_hyperedges = len(hypergraph)  # <-- Move this line here!
    hyperedge_index = []
    for he_idx, nodes in enumerate(hypergraph):
        hyperedge_index.extend([[n, he_idx] for n in nodes])
    hyperedge_index = torch.tensor(hyperedge_index).t().contiguous()
    #edge_index = hypergraph_to_edge_index(hypergraph)
    data=Data(x=features, edge_index=hyperedge_index, y=labels,num_hyperedges=num_hyperedges)
    return data,unique_labels



def main(args):
    #print(args)
    import torch_geometric.transforms as T
    data,unique_classes = dataset_preprocess(args.dataset,args.types)
    y = data.y.detach().cpu().numpy()
    idx = np.array(range(len(y)))
    np.random.seed(args.data_seed)
    np.random.shuffle(idx)
    # Build a fixed train/validation/calibration split from user-controlled fractions.
    train_end = int(args.train_fraction * len(idx))
    valid_end = int((args.train_fraction + args.valid_fraction) * len(idx))
    split_res = np.split(idx, [train_end, valid_end, len(idx)])
    train_idx, valid, calib_test = split_res[0], split_res[1], split_res[2]

    train_mask = np.array([False] * len(y))
    train_mask[train_idx] = True

    valid_mask = np.array([False] * len(y))
    valid_mask[valid] = True

    calib_test_mask = np.array([False] * len(y))
    calib_test_mask[calib_test] = True
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    data.calib_test_mask = torch.tensor(calib_test_mask, dtype=torch.bool)
    n = min(args.max_calib_size, int(calib_test.shape[0]/2))
    alpha = args.alpha
    tau = args.tau
    target_size = args.target_size
    num_conf_layers = args.confgnn_num_layers
    base_model = args.confgnn_base_model
    tau2res = {}

    for run in tqdm(range(args.num_runs)):
        result_this_run = {}
        if args.quantile:
            if args.alpha == 0.1:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_0410.pt'
            else:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_' + str(args.alpha) + '_0410.pt'
        else:
            model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_0410.pt'

        output_dim = unique_classes
        num_features = data.x.shape[1]
        if (os.path.exists(model_checkpoint)) and (not args.retrain):
            print('loading saved base model...',flush=True)
            model = torch.load(model_checkpoint, map_location = device)
            model, data = model.to(device), data.to(device)
            model.eval()
            pred = model(data.x, data.edge_index)
            best_model = model
            best_pred = pred
        else:
            print('training base model from scratch...',flush=True)
            model = HGNN(num_features, output_dim, args.hidden_channels, dropout=args.base_dropout)

            model, data = model.to(device), data.to(device)
            sampler=edge_task(data,embed_dim=args.hidden_channels)
            auxillary_model=HyperedgeDegreePredictor(in_features=output_dim,hidden_dim=args.hidden_channels)
            sampler, auxillary_model = sampler.to(device), auxillary_model.to(device)
            optimizer = torch.optim.Adam(list(model.parameters()),lr=args.lr)

            best_val_acc = final_test_acc = 0
            for epoch in range(1, args.epochs + 1):
                loss = train(epoch, model, data, optimizer, alpha)
                (train_acc, val_acc, tmp_test_calib_acc), pred = test(model, data, alpha, tau, target_size)
                if val_acc > best_val_acc:
                    #torch.save(best_model, model_checkpoint)
                    best_model = copy.deepcopy(model)
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                if args.verbose:
                    if epoch % args.log_interval == 0:
                        print('Epoch:', epoch, 'Loss:', loss, 'Train:', train_acc, 'Val:', val_acc)

            (train_acc, val_acc, test_acc), _ = test(best_model, data, alpha, tau, target_size, size_loss = False)
            print('Final test accuracy:', test_acc,flush=True)
            print('Final validation accuracy:', val_acc,flush=True)
            print('Final training accuracy:', train_acc,flush=True)
            model_to_correct = copy.deepcopy(model)
            if args.conf_correct_model == 'hnn':
                confmodel = ConfHNN(model_to_correct, data, args, num_conf_layers, output_dim).to(args.device)
            elif args.conf_correct_model == 'mlp':
                confmodel = ConfMLP(model_to_correct, data, args, output_dim).to(args.device)
            optimizer = torch.optim.Adam(list(confmodel.parameters())+list(sampler.parameters())+list(auxillary_model.parameters()), weight_decay=args.confgnn_weight_decay, lr=args.confgnn_lr)
            best_size_loss = float('inf')
            best_val_acc = 0
            # Split calibration/test nodes again when a calibration holdout is requested.
            calib_test_idx = np.where(data.calib_test_mask.detach().cpu().numpy())[0]
            np.random.seed(run)
            np.random.shuffle(calib_test_idx)
            calib_eval_idx = calib_test_idx[:int(n * args.calib_fraction)]
            calib_test_real_idx = calib_test_idx[int(n * args.calib_fraction):]

            data.calib_eval_mask = np.array([False] * len(y))
            data.calib_eval_mask[calib_eval_idx] = True
            data.calib_test_real_mask = np.array([False] * len(y))
            data.calib_test_real_mask[calib_test_real_idx] = True
            calib_eval_idx = np.where(data.calib_eval_mask)[0]
            np.random.seed(run)
            np.random.shuffle(calib_eval_idx)
            train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
            train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
            train_train_idx = np.where(data.train_mask.detach().cpu().numpy())[0]

            print('Starting topology-aware conformal correction...',flush=True)
            for epoch in range(1, args.epochs + 1):
                confmodel.train()
                sampler.train()
                auxillary_model.train()
                optimizer.zero_grad()
                # Select the most informative hyperedges for the auxiliary topology loss.
                sampler.attention()
                topk_mask,L=sampler(args.edge_topk)
                # disable gradient for L
                L.requires_grad = False
                #true_degrees = (L.to_dense() != 0).sum(dim=1).float()
                true_degrees = torch.bincount(L._indices()[0], minlength=L.size(0)).float()

                out1,out2, ori_out = confmodel(data.x, data.edge_index)
                # create hyperedge features by taking mean of the features of the nodes (ori_out) in the hyperedge

                hyperedge_features = scatter_mean(
                    ori_out[data.edge_index[0]],
                    data.edge_index[1],
                    dim=0,
                    dim_size=data.num_hyperedges
                )
                hyperedge_features = hyperedge_features.to(device)
                # enable gradient for hyperedge features
                hyperedge_features.requires_grad = True
                selected_degrees = hyperedge_features * topk_mask.unsqueeze(1)
                selected_true_degrees = true_degrees * topk_mask
                pred_degree = auxillary_model(selected_degrees).squeeze(-1)  # [62481]
                degree_loss = F.mse_loss(pred_degree, selected_true_degrees)
                out_softmax1 = F.softmax(out1, dim = 1)
                out_softmax2 = F.softmax(out2, dim = 1)
                out_softmx_mean = (out_softmax1 + out_softmax2)/2
                out_mean= (out1 + out2)/2
                ori_out_softmax = F.softmax(ori_out, dim = 1)

                n_temp = len(train_calib_idx)
                q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp
                # RAPS-style score used to optimize set size during correction.
                lam_reg = args.raps_lam_reg
                C = out_softmx_mean.shape[1]
                k_reg = max(0, min(args.raps_k, C - 1))  # ensure k_reg < C
                dev = out_softmx_mean.device

                reg_vec = torch.cat([
                    torch.zeros(k_reg, device=dev),
                    torch.full((C - k_reg,), lam_reg, device=dev)
                ]).unsqueeze(0)  # [1, C]

                # Get calibration set outputs and true labels
                cal_out = out_softmx_mean[train_calib_idx]        # [n_temp, num_classes]
                cal_labels = data.y[train_calib_idx]              # [n_temp]

                # Sort probabilities (descending)
                cal_pi = torch.argsort(cal_out, dim=1, descending=True)  # [n_temp, num_classes]
                cal_srt = torch.gather(cal_out, 1, cal_pi)               # [n_temp, num_classes]
                cal_srt_reg = cal_srt + reg_vec                         # [n_temp, num_classes]

                # Find location of true label in the sorted index
                cal_L = (cal_pi == cal_labels.unsqueeze(1)).nonzero()[:,1]  # [n_temp] label positions for each example

                # Cumulative sum up to true label
                cal_cumsum = torch.cumsum(cal_srt_reg, dim=1)              # [n_temp, num_classes]
                cal_score = cal_cumsum[torch.arange(n_temp), cal_L]        # [n_temp]

                # Randomized subtraction (RAPS): subtract random fraction of regularized score at true label
                rand_uniform = torch.rand(n_temp, device=dev)
                #import pdb;pdb.set_trace()
                cal_score = cal_score - rand_uniform * cal_srt_reg[torch.arange(n_temp), cal_L]  # [n_temp]

                # Now you have RAPS-style conformal scores in cal_score (torch tensor)
                tps_conformal_score = cal_score
                #tps_conformal_score = out_softmx_mean[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
                qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')

                c = torch.sigmoid((out_softmx_mean[train_test_idx] - qhat)/tau)
                size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))
                if args.cond_cov_loss:
                    ## coverage loss
                    unique_classes = torch.unique(data.y)
                    y = data.y[train_test_idx]
                    loss_cov = torch.zeros(1).to(device)
                    for i in unique_classes:
                        class_mask = y == i
                        loss_cov += -torch.mean(c[torch.arange(c.shape[0]), y][class_mask])
                    loss_cov = (1/len(unique_classes)) * loss_cov

                    loss_cov = loss_cov.squeeze()
                    #print(loss_cov.item())
                pred_loss = F.cross_entropy(out_mean[train_train_idx], data.y[train_train_idx])
                cont_loss= InfoNCE(out1[train_train_idx], out2[train_train_idx],T=args.temperature)
                if args.conftr:
                    if epoch <= args.warmup_epochs:
                        loss = pred_loss + size_loss
                    elif args.cond_cov_loss:
                        if epoch <= args.coverage_warmup_epochs:
                            loss = pred_loss + args.size_loss_weight * size_loss + args.contrastive_loss_weight * cont_loss + degree_loss
                        else:
                            loss = pred_loss + args.size_loss_weight * size_loss + loss_cov + args.contrastive_loss_weight * cont_loss + degree_loss
                    else:
                        loss = pred_loss + args.size_loss_weight * size_loss + degree_loss + args.late_contrastive_loss_weight * cont_loss
                else:
                    loss = pred_loss
                loss.backward()
                #print(epoch)
                optimizer.step()
                if args.verbose:
                    if epoch % args.log_interval == 0:
                        print('Epoch:', epoch, 'Loss:', loss.item(), 'Size Loss:', size_loss.item(), 'Pred Loss:', pred_loss.item(), 'Cont Loss:', cont_loss.item(), 'Degree Loss:', degree_loss.item())

            (train_acc, val_acc, tmp_test_calib_acc), pred, size_loss = test(confmodel, data, alpha, tau, target_size, size_loss = True)
            eff_valid = run_conformal_classification(pred, data, n, alpha, score = 'aps', validation_set = True, conformal_trials=args.conformal_trials, raps_lam_reg=args.raps_lam_reg, raps_k=args.raps_k)[1]
            if args.conftr:
                if eff_valid < best_size_loss:
                    best_size_loss = eff_valid
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                    best_epoch = epoch
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
            result_this_run['cont_conf_hgnn'] = {}
            result_this_run['cont_conf_hgnn']['APS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction, conformal_trials=args.conformal_trials, raps_lam_reg=args.raps_lam_reg, raps_k=args.raps_k)
            result_this_run['cont_conf_hgnn']['RAPS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction, conformal_trials=args.conformal_trials, raps_lam_reg=args.raps_lam_reg, raps_k=args.raps_k)
            result_this_run['cont_conf_hgnn']['eff_valid'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', validation_set = True, conformal_trials=args.conformal_trials, raps_lam_reg=args.raps_lam_reg, raps_k=args.raps_k)[1]
            result_this_run['cont_conf_hgnn']['eff_valid_raps'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', validation_set = True, conformal_trials=args.conformal_trials, raps_lam_reg=args.raps_lam_reg, raps_k=args.raps_k)[1]

            print(result_this_run,flush=True)
            tau2res[run] = result_this_run
            print('Finished training this run!')
main(args)
