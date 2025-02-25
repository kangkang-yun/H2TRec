import pickle
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
from center_loss import CenterLoss
import torch.utils.data as data
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA
import torch.nn.functional as F
import scipy.sparse as sp

class DictData(data.Dataset):
    def __init__(self, train_raing_dict=None, is_training=None, data_set_count=0):
        super(DictData, self).__init__()

        self.train_raing_dict = train_raing_dict
        self.is_training = is_training
        self.data_set_count = data_set_count
        # 构造这个类的对象时需要三个参数

    def __len__(self):  # 获取数据集长度
        return self.data_set_count  # return self.num_ng*len(self.train_dict)

    def __getitem__(self, idx):  # 获取给定索引的项目，包括特征用户，标签，物品
        features = self.train_raing_dict
        user = features[idx][0]
        label_r = np.array(features[idx][2])  # 创建为一个数组类型ml1m需要+1
        item = features[idx][1]
        return user, item, label_r.astype(np.float32) # float32  .astype(np.int)

dataset_name = 'Ciao'                # dataset: Epinions, Ciao or yelp
dataset_path = '../dataset/' + dataset_name + '/'


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


set_seed(30000)


def picture_loss(pic_list, curve_name):
    x_major_locator = MultipleLocator(1)
    x_label = range(len(pic_list))
    y_label = pic_list
    plt.figure()

    ax = plt.axes()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(x_label, y_label, linewidth=1, linestyle='solid')
    plt.legend()
    plt.title('%s curve' % curve_name)
    plt.show()


def paint_subgraph(hyper_graph, emb_users, emb_items, epoch_count):
    pca = PCA(n_components=2)
    colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple',
              'xkcd:brown', 'xkcd:grey', 'xkcd:yellow', 'xkcd:pink', 'xkcd:teal']
    col_rand_array = np.arange(hyper_graph.shape[1])
    np.random.shuffle(col_rand_array)
    col_rand = hyper_graph[:, col_rand_array[0:10]]
    label_list = []
    for i in range(0, 10):
        label_list.append(torch.tensor(np.argwhere(col_rand[:, i] == 1).flatten()).cuda())
    all_emb = torch.cat([emb_users, emb_items])
    for i in range(0, 10):
        paint_emb = F.embedding(label_list[i], all_emb)
        paint_arr = np.array(paint_emb.cpu().detach())
        compress_embedding = pca.fit_transform(paint_arr)
        plt.scatter(compress_embedding[:, 0], compress_embedding[:, 1], s=1, c=colors[i])
    plt.title("node distribution(epoch %d)" % epoch_count)
    plt.savefig("./Ciao/epoch %d.png" % epoch_count)
    plt.show()

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


with open(dataset_path + 'dataset_filter5.pkl', 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)


with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_users_items_list = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)
    u_fans_list = pickle.load(f)
    u_fans_items_list = pickle.load(f)

hyper_martix = np.load('./centlabel.npy', allow_pickle=True)
hyper_graph_1 = sp.load_npz('./hyper_graph_1.npz')
hyper_graph_2 = sp.load_npz('./hyper_graph_2.npz')
pre_adj = sp.load_npz('./s_pre_adj_mat.npz')
hypergraph_1 = _convert_sp_mat_to_sp_tensor(hyper_graph_1)
hypergraph_2 = _convert_sp_mat_to_sp_tensor(hyper_graph_2)
global_graph = _convert_sp_mat_to_sp_tensor(pre_adj)

train_dict = {}
for index, sample in enumerate(train_set):
    uid = sample[0]
    iid = sample[1]
    score = sample[2]
    train_dict[index] = [uid, iid, score]

train_set = DictData(train_raing_dict=train_dict, is_training=True, data_set_count=len(train_dict))
train_loader = DataLoader(train_set, batch_size=2048, shuffle=True, num_workers=0)

all_epoch = 30
dim = 80
num_classes = hyper_martix.shape[1]


model = CenterLoss(num_classes, dim, hyper_martix, user_count, item_count, hypergraph_1, hypergraph_2, global_graph)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
min_loss = 1e15
loss_list = []
for epoch in range(all_epoch):
    print('=====================================')
    print(f'EPOCH{epoch}/{all_epoch}')
    all_loss = 0
    for user_batch, item_batch, rating_batch in tqdm(train_loader):
        user_batch = user_batch.cuda()
        item_batch = item_batch.cuda()
        loss = model(user_batch, item_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

    current_loss = math.sqrt(all_loss / (user_count + item_count))
    if current_loss < min_loss:
        print('current loss, min_loss: ', current_loss, min_loss)
        min_loss = current_loss
        loss_list.append(min_loss)

    # paint_subgraph(hyper_martix, model.user_emb, model.item_emb, epoch)

# picture_loss(loss_list, 'center-loss')
user_emb = model.user_emb.detach().cpu().numpy()
item_emb = model.item_emb.detach().cpu().numpy()

with open('%s/center_pretrain_set_model_%s_weights.pkl' % (dataset_name, dim), 'wb') as f:
    pickle.dump(user_emb, f)
    pickle.dump(item_emb, f)
