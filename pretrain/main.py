#remove some unnecessary codes
import world
from world import cprint
import utils
from model import LightGCN
import dataloader
import torch
import numpy as np
import time
import os
import Procedure
import dataloader
from torch.utils.data import DataLoader
from os.path import join
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
utils.set_seed(30000)
dataset = dataloader.Loader(path="../data/"+world.dataset)
Recmodel = LightGCN(world.config, dataset) #initialize MODEL
Recmodel = Recmodel.to(world.device) #put model into GPU
bpr = utils.BPRLoss(Recmodel, world.config) #initialize Optim
mse = utils.MSELoss(Recmodel, world.config, dataset)
#cen = utils.CENTERLoss(Recmodel, world.config, dataset)

batch_size = 2048
ndcg = 0
best_result = {}
train_set = np.load('../data/'+world.dataset+'/training_ratings_dict.npy', allow_pickle=True).item()
train_set = utils.DictData(train_raing_dict=train_set, is_training=True, data_set_count=len(train_set))
train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True, num_workers=0)
best_rmse = 5
for epoch in range(world.TRAIN_epochs):
    print('======================')
    print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
    start = time.time()
    if epoch % 1 == 0:
        # cprint("[TEST]")
        result, best_rmse = Procedure.Test(dataset, Recmodel, best_rmse, world.config['multicore'])
        # f = open('lightGCN_{}_layer{}_result.txt'.format(world.dataset, world.config['lightGCN_n_layers']), 'a')
        # f.write('epoch = {} \n'.format(epoch))
        # f.write('result = {} \n'.format(result))
        # f.close()
        # if result['ndcg'] > ndcg:
        #     ndcg = result['ndcg']
        #     best_epoch = epoch
        #     best_result = result
    output_information = Procedure.MSE_train_original(train_loader, Recmodel, mse)
    # if epoch % 2 == 1:
    #     cprint("[TEST]")
    #     result = Procedure.Test(dataset, Recmodel, world.config['multicore'])
    #     output_information = Procedure.MSE_train_original(train_loader, Recmodel, cen)
    #output_information = Procedure.MSE_train_original(train_loader, Recmodel, mse)
    # ***************paint start***************
    # user_e, item_e = Recmodel.computer()
    # all_emb = torch.cat([user_e, item_e])
    # label = np.load('../data/filmtrust/centlabel_100.npy', allow_pickle=True)     # node * group
    # i = 0
    # label_list = []
    # for i in range(0, Recmodel.num_groups):
    #     label_list.append(torch.tensor(np.argwhere(label[:, i] == 1).flatten()).to(world.device))
    #
    # # 初始化 PCA 降维模型
    # pca = PCA(n_components=2)
    #
    # # 可视化
    # colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple',
    #           'xkcd:brown', 'xkcd:grey', 'xkcd:yellow', 'xkcd:pink', 'xkcd:teal']
    # i = 0
    # center_arr = np.array(mse.center_loss.centers.cpu().detach())
    # compress_center_embedding = pca.fit_transform(center_arr)
    # for i in range(0, Recmodel.num_groups):
    #     plt.scatter(compress_center_embedding[i, 0], compress_center_embedding[i, 1],
    #                 s=200, c=colors[i], marker='*', edgecolors='black')
    #     paint_emb = F.embedding(label_list[i], all_emb)
    #     paint_arr = np.array(paint_emb.cpu().detach())
    #     compress_embedding = pca.fit_transform(paint_arr)
    #     plt.scatter(compress_embedding[:, 0], compress_embedding[:, 1], s=1, c=colors[i])
    # plt.title("node distribution(epoch {index})".format(index=epoch))
    # plt.savefig("../embedding可视化/epoch {index}.png".format(index=epoch))
    # plt.show()
    # ***************paint end***************
    print(f"[TOTAL TIME] {time.time() - start}")

# f = open('lightGCN_{}_layer{}_result.txt'.format(world.dataset, world.config['lightGCN_n_layers']), 'a')
# f.write('best_epoch = {} \n'.format(best_epoch))
# f.write('best_result = {} \n'.format(best_result))
# f.close()
#
user_emb = Recmodel.embedding_user.weight.detach().cpu().numpy()
item_emb = Recmodel.embedding_item.weight.detach().cpu().numpy()

# with open('center_pretrain_set_model_80_weights_ciao.pkl', 'wb') as f:
#     pickle.dump(user_emb, f)
#     pickle.dump(item_emb, f)

# with open('../data/'+world.dataset+'/center_pretrain_set_model_80_weights_random.pkl', 'wb') as f:
#     pickle.dump(user_emb, f)
#     pickle.dump(item_emb, f)
