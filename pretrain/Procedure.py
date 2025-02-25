import json

import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from center_loss import CenterLoss


CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class
    S, sam_time = utils.UniformSample_original(dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"

def MSE_train_original(train_loader, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    mseloss = loss_class
    for user_batch, item_batch, rating_batch in train_loader:
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        mseloss.stageOne(user_batch, rating_batch, item_batch)


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs(predictions - targets).mean()

def Test(dataset, Recmodel, best_rmse, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    testDict= dataset.testDict   # 将测试集用户和物品做成字典
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    # results = {'precision': np.zeros(len(world.topks)),             # len(world.topks)=1
    #            'recall': np.zeros(len(world.topks)),
    #            'ndcg': np.zeros(len(world.topks))}
    results = {}
    with torch.no_grad():
        # users = list(testDict.keys())                     # 测试集用户集合，list形式
        # test_items = list(set(dataset.testItem)) #set the testitem list only in test dataset
        # allitem_list = list(range(dataset.m_items))
        # outitems = list(set(allitem_list) - set(test_items))  # 不在测试集的物品
        # auc_record = []
        # # ratings = []
        # total_batch = len(users) // u_batch_size + 1
        # allPos = dataset.getUserPosItems(users)     # 训练集中测试集用户交互过的商品
        # groundTrue = [testDict[u] for u in users]       # 测试集每一个用户测试物品的集合
        # users_gpu = torch.Tensor(users).long()
        # users_gpu = users_gpu.to(world.device)
        #
        # rating = Recmodel.getUsersRating(users_gpu)   #得到预测的评分信息
        # #rating = rating.cpu()
        # exclude_index = []
        # exclude_items = []
        # for range_i, items in enumerate(allPos):
        #     exclude_index.extend([range_i] * len(items))            #不推荐已经表现积极的物品
        #     exclude_items.extend(items)
        #     exclude_index.extend([range_i] * len(outitems))         #不推荐不在测试集的物品
        #     exclude_items.extend(outitems)
        # rating[exclude_index, exclude_items] = -(1<<10)
        # _, rating_K = torch.topk(rating, k=max_K)
        # rating = rating.cpu().numpy()
        # aucs = [
        #         utils.AUC(rating[i],
        #                   dataset,
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        # del rating
        # x = [rating_K.cpu(), groundTrue]
        # result = test_one_batch(x)
        #
        # results['recall'] += result['recall']
        # results['precision'] += result['precision']
        # results['ndcg'] += result['ndcg']
        # results['recall'] /= float(len(users))
        # results['precision'] /= float(len(users))
        # results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        user_embed, item_embed = Recmodel.computer()
        user_embed = user_embed.cpu().detach().numpy()
        item_embed = item_embed.cpu().detach().numpy()
        test_rating_dict = np.load('../data/'+world.dataset+'/testing_ratings_dict.npy',
                                               allow_pickle=True).item()
        pre_all = []
        label_all = []
        res = []
        for pair_i in test_rating_dict:
            u_id, i_id, r_v = test_rating_dict[pair_i]
            # r_v += 1
            pre_get = np.sum(user_embed[u_id] * item_embed[i_id])
            pre_all.append(pre_get)
            label_all.append(r_v)
            res.append([u_id, i_id, r_v, pre_get])
        rmse_test = rmse(np.array(pre_all), np.array(label_all))
        mae_test = mae(np.array(pre_all), np.array(label_all))
        res_test = round(np.mean(rmse_test), 4)
        mae_test = round(np.mean(mae_test), 4)
        results['RMSE'] = res_test
        results['MAE'] = mae_test
        print(results)
        # if rmse_test < best_rmse:
        #     best_rmse = rmse_test
        #     with open("../result/%s/wide_test_predict.txt" % world.dataset, 'w') as f:
        #         f.truncate(0)
        #         for i in res:
        #             context = '{},{},{},{}'.format(i[0], i[1], i[2], i[3])
        #             f.write('\n')
        #             f.write(context)
        return results, best_rmse
