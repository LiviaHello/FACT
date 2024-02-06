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

from utils import *
from models import *
from tqdm import tqdm

import scipy.stats as stats

import pdb
import sys

# Had to edit some of the functions to work on our pickle file since we are 
# unsure how they created their pickle file from the original datasets

# These are from the metric.py function
def recall(ranked_list, ground_list):
    if not ground_list:  # Check if ground_list is empty
        return 0.0  # Return a default value, like 0, in this case

    hits = 0
    for id in ranked_list:
        if id in ground_list:
            hits += 1
    rec = hits / float(len(ground_list))
    return rec


def ndcg(ranked_list, ground_truth):
    if not ground_truth:  # Check if ground_truth is empty
        return 0.0  # Return a default value, like 0, in this case

    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def js_topk(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return rank_js_distance, truth_rank_js_distance

# these are from the loss.py file
def bpr_loss_here(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    mf_loss = torch.mean(F.softplus(neg_score - pos_score))
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_bpr_loss_here(user_emb, item_emb, u, i, j):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    batch_neg_item_emb = item_emb[j]

    mf_loss, emb_loss = bpr_loss_here(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
    return mf_loss, emb_loss

# the below is the adapted from train_bpr_baseline_lastfm.py
def train_bprmf_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
        }
        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)

            main_user_emb, main_item_emb = model.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg
            loss = bpr_loss + emb_loss

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr_loss'] += bpr_loss.item()
            train_res['emb_loss'] += emb_loss.item()

        train_res['bpr_loss'] = train_res['bpr_loss'] / len(train_loader)
        train_res['emb_loss'] = train_res['emb_loss'] / len(train_loader)

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = model.forward()
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
        description='lastfm_bpr_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='bpr')
    parser.add_argument('--dataset', type=str, default='./data/marketBias/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/bpr_base_electronics.txt')
    parser.add_argument('--param_path', type=str, default='param/bpr_base_electronics.pth')
    parser.add_argument('--num_epochs', type=int, default=255) # we shortened the number of epochs due to computational cost
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

    bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    train_bprmf_baseline(bprmf, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args)
    sys.stdout = None
