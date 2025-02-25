
import world
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from time import time
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from center_loss import CenterLoss


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

class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss *self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class MSELoss:
    def __init__(self, recmodel, config, dataset):
        self.model = recmodel
        self.lr = config['lr']
        self.dataset = dataset
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss().cuda()
        self.TrustNet = self.dataset.TrustNetWork
        self.UserItemNet = self.dataset.UserItemNetWork
        self.users = self.dataset.n_users

        # center_loss
        self.center_loss = CenterLoss(num_classes=world.sub_num, feat_dim=config['latent_dim_rec'], dataset=dataset, use_gpu=True)
        params = list(recmodel.parameters()) + list(self.center_loss.parameters())
        self.opt = optim.Adam(params, lr=self.lr)
        # self.alpha =
        self.beta = 0.1


    def stageOne(self, user_batch, rating_batch, item_batch):
        user_e, item_e = self.model.computer()
        user_b = F.embedding(user_batch, user_e)
        item_b = F.embedding(item_batch, item_e)
        prediction = (user_b * item_b).sum(dim=-1)
        loss_part = self.mse_loss(prediction, rating_batch)
        l2_regulization = 0.0001 * (user_b ** 2 + item_b ** 2).sum(dim=-1)#正则化项系数，ml1m=0.01，lastfm=0.0001
        #l2_regulization1 = 0.0001 * (group_e ** 2).sum(dim=-1)

        item_batch_temp = item_batch + self.users
        centloss1, size1 = self.center_loss(user_b, user_batch)
        centloss2, size2 = self.center_loss(item_b, item_batch_temp)
        #print(self.beta*(centloss1 + centloss2)/(size1 + size2))
        loss = loss_part + l2_regulization.mean() + self.beta*(centloss1 + centloss2)/(size1 + size2)
        self.opt.zero_grad()
        loss.backward()
        # for param in self.center_loss.parameters():
        #     # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
        #     param.grad.data *= (1 / (self.beta * self.lr))
        self.opt.step()
        return loss.cpu().item()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class CENTERLoss:
    def __init__(self, recmodel, config, dataset):
        self.model = recmodel
        self.lr = 0.01
        self.dataset = dataset
        self.center_loss = CenterLoss(num_classes=100, feat_dim=config['latent_dim_rec'], dataset=dataset, use_gpu=True)
        params = list(recmodel.parameters()) + list(self.center_loss.parameters())
        self.opt = optim.Adam(params, lr=self.lr)
        self.beta = 0.01
        self.users = self.dataset.n_users

    def stageOne(self, user_batch, rating_batch, item_batch):
        user_e, item_e = self.model.computer()
        user_b = F.embedding(user_batch, user_e)
        item_b = F.embedding(item_batch, item_e)
        l2_regulization = 0.001 * (user_b ** 2 + item_b ** 2).sum(dim=-1)
        item_batch_temp = item_batch + self.users
        centloss1, size1 = self.center_loss(user_b, user_batch)
        centloss2, size2 = self.center_loss(item_b, item_batch_temp)
        loss = self.beta*(centloss1 + centloss2)/(size1 + size2) + l2_regulization.mean()
        self.opt.zero_grad()
        loss.backward()
        for param in self.center_loss.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            param.grad.data *= (0.05 / (self.beta * self.lr))
        self.opt.step()
        return loss.cpu().item()


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def UniformSample_original(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n

    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')