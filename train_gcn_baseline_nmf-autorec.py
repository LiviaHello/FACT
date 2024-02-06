# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF

from utils import *
from models import *
from tqdm import tqdm



import scipy.stats as stats

import pdb
import sys


def train_gcn_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_AutoRec = optim.Adam(autorec.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # For AutoRec

    # Initialize matrices
    user_item_matrix = torch.zeros(n_users, n_items)
    mask_matrix = torch.zeros(n_users, n_items)

    # Iterate through the train_u2i
    for user, item_list in train_u2i.items():
      for item in item_list:
        # Check if the item is an integer (item ID)
        if isinstance(item, int):
            user_item_matrix[user, item] = 1  # Binary so 1 is the rating assignment
            mask_matrix[user, item] = 1
        else:
            print(f"Unexpected format for item: {item}")

    # NMF implementation
    n_components = 20
    nmf_model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    nmf_user_features = nmf_model.fit_transform(user_item_matrix)  # User latent features
    nmf_item_features = nmf_model.components_.T  # Item latent features

    # Convert to PyTorch tensors and send to device
    nmf_user_features_tensor = torch.tensor(nmf_user_features).float().to(args.device)
    nmf_item_features_tensor = torch.tensor(nmf_item_features).float().to(args.device)

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'autorec_loss': 0.0,
        }
        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)

            main_user_emb, main_item_emb = model.forward()

            # Concatenate NMF features with the embeddings from interest encoder
            main_user_emb = torch.cat([main_user_emb, nmf_user_features_tensor], dim=1)
            main_item_emb = torch.cat([main_item_emb, nmf_item_features_tensor], dim=1)

            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg
            loss = bpr_loss + emb_loss

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr_loss'] += bpr_loss.item()
            train_res['emb_loss'] += emb_loss.item()

        # Outside the trainloader loop because it is for the entire dataset
        autorec_loss = autorec.train_model(user_item_matrix.to(args.device), mask_matrix.to(args.device), optimizer_AutoRec, args.device)
        train_res['autorec_loss'] += autorec_loss.item()

        train_res['bpr_loss'] = train_res['bpr_loss'] / len(train_loader)
        train_res['emb_loss'] = train_res['emb_loss'] / len(train_loader)
        train_res['autorec_loss'] =  train_res['autorec_loss'] / len(train_loader)

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = model.forward()
            t_user_emb = torch.cat([t_user_emb, nmf_user_features_tensor], dim=1)
            t_item_emb = torch.cat([t_item_emb, nmf_item_features_tensor], dim=1)
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens,
                num_workers=args.num_workers)

            p_eval = ''
            for keys, values in test_res.items():
                p_eval += keys + ':' + '[%.6f]' % values + ' '
            print(p_eval)

            if best_perf < test_res['ndcg@10']:
                best_perf = test_res['ndcg@10']
                torch.save(model, args.param_path)
                print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='gcn')
    #parser.add_argument('--dataset', type=str, default='/content/drive/My Drive/data_fairmi/ml-1m/process/process.pkl') #for colab
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--log_path', type=str, default='logs/gcn_base_nmf-autorec.txt')
    #parser.add_argument('--param_path', type=str, default='/content/drive/My Drive/param_fairmi/gcn_base.pth') # for colab
    parser.add_argument('--param_path', type=str, default='param/gcn_base.pth')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    sys.stdout = Logger(args.log_path)
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    # bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    autorec = AutoRec(n_items, n_users, args.emb_size, args.emb_size, args.device)
    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    gcn = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    train_gcn_baseline(gcn, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args)
    sys.stdout = None
