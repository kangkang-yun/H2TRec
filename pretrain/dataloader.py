
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from math import e
from scipy.sparse.linalg import gmres
from scipy.sparse.csgraph import shortest_path
# from sparse_dot_mkl import dot_product_mkl
import math
from tqdm import tqdm

class Loader():
    """
    Dataset type for pytorch \n
    Incldue graph information

    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.max_score = 0
        self.min_score = 10
        self.top_k = world.sub_num  # 选取k个最值得信任的用户
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        trust_file = path + '/trust.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser, trainScore = [], [], [], []
        testUniqueUsers, testItem, testUser, testScore = [], [], [], []
        truster, trustee = [], []
        TrustDegree = []
        self.traindataSize = 0
        self.testDataSize = 0
        self.trustsize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = int(l[1])
                    uid = int(l[0])
                    score = float(l[2])
                    trainUniqueUsers.append(uid)
                    trainUser.append(uid)
                    trainItem.append(items)
                    trainScore.append(score)
                    self.m_item = max(self.m_item, items)
                    self.n_user = max(self.n_user, uid)
                    self.max_score = max(self.max_score, score)
                    self.min_score = min(self.min_score, score)
                    self.traindataSize += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainScore = np.array(trainScore)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = int(l[1])
                    uid = int(l[0])
                    score = float(l[2])
                    testUniqueUsers.append(uid)
                    testUser.append(uid)
                    testItem.append(items)
                    testScore.append(score)
                    self.m_item = max(self.m_item, items)
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += 1
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.testScore = np.array(testScore)


        with open(trust_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    truster_id = int(l[0])
                    trustee_id = int(l[1])
                    if trustee_id >= self.n_user or truster_id >= self.n_user:
                        continue
                    truster.append(truster_id)
                    trustee.append(trustee_id)
                    self.trustsize += 1
        self.truster = np.array(truster)
        self.trust_node_number = len(np.unique(self.truster))
        self.trustee = np.array(trustee)

        self.Graph = None
        self.TrustGraph = None
        self.TrustDegreeB = None
        self.TrustDegreeB_improve = None
        self.hypergraph = None
        self.hypergraph_1 = None
        self.hypergraph_2 = None
        self.fusion_graph = None
        self.TrustTopuGraph = None
        self.centlabel = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.trustsize} interactions for trust information")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        print(f"{world.dataset} Trust Sparsity : {self.trustsize / self.n_users / self.n_users}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))     # 压缩后形状为 用户数*电影数
        self.UserItemRatings = csr_matrix((self.trainScore, (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.TrustNet = csr_matrix((np.ones(len(self.truster)), (self.truster, self.trustee)),
                                    shape=(self.n_user, self.n_user))
        # print(self.UserItemRatings[10].mean())

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self.user_rating_sum = np.array(self.UserItemRatings.sum(axis=1)).squeeze()
        self.user_rating_avg = self.user_rating_sum / self.users_D

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def topks(self):
        return self.top_k

    @property
    def TrustNetWork(self):
        return self.TrustNet

    @property
    def UserItemNetWork(self):
        return self.UserItemNet

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _convert_array_to_csr_matrix(self, X):
        row, col = np.nonzero(X)
        values = X[row, col]
        csr_x = csr_matrix((values, (row, col)), shape=(X.shape[0], X.shape[1]))
        return csr_x

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
                self.global_graph = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemRatings.tolil()
                adj_mat[:self.n_users, self.n_users:] = R               # 放在右上角
                adj_mat[self.n_users:, :self.n_users] = R.T             # 放在左下角
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                self.global_graph = adj_mat
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)                          # 得到拉普拉斯矩阵
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph


    def getTrustnetwork(self):
        print("loading trust matrix")
        if self.TrustGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/trust.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating trust matrix")
                adj_mat = self.TrustNet
                adj_mat = adj_mat.tolil()
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/trust.npz', norm_adj)
            if self.split == True:
                self.TrustGraph = self._split_A_hat(norm_adj)
            else:
                self.TrustGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.TrustGraph = self.TrustGraph.coalesce().to(world.device)

        return self.TrustGraph


    def calpcc(self, user1, user2, user1_id, user2_id):   #pcc 映射在[-1,1]之间
        user1 = self._convert_sp_mat_to_sp_tensor(user1)
        column1 = user1._indices().numpy()[1]
        value1 = user1._values().numpy()
        user2 = self._convert_sp_mat_to_sp_tensor(user2)
        column2 = user2._indices().numpy()[1]
        value2 = user2._values().numpy()
        common_item = set(column1).intersection(set(column2))
        sub = 0.0
        if common_item:
            for item in common_item:
                #index_1 = column1.index(item)
                index_1 = np.where(column1 == item)
                index_2 = np.where(column2 == item)
                sub += np.absolute(value1[index_1] - value2[index_2])
            avg_sub = sub / len(common_item)
            result = e ** (-avg_sub)
            # result = sum_1 / sum_2 ** 0.5 / sum_3 ** 0.5
        else:
            result = np.array([e ** -3.5])        #差值分布在[0,3.5]之间
            # result = np.array([-1.0])
        #return 2 - 4 / (result + 3)
        return 2 - 2 / (result + 1)

    def get_degree_martix(self):
        if self.TrustDegreeB is None:
            try:
                trust_degree = np.load(self.path + '/temp_trust_degree.npy')
                self.trustdegree = self._convert_array_to_csr_matrix(trust_degree)
            except:
                UserItemRatings_temp = self.UserItemRatings.toarray()
                pcc_matrix = np.corrcoef(UserItemRatings_temp)
                pcc_matrix = pcc_matrix * 0.5 + 0.5
                np.fill_diagonal(pcc_matrix, 0)
                pcc_matrix[np.isnan(pcc_matrix)] = 0
                trust_degree = np.divide(np.multiply(2 * pcc_matrix, self.TrustNet.toarray()),
                                         np.add(pcc_matrix, self.TrustNet.toarray()))
                trust_degree[np.isnan(trust_degree)] = 0

                self.trustdegree = self._convert_array_to_csr_matrix(trust_degree)
                np.save(self.path + '/temp_trust_degree.npy', trust_degree)


            try:
                pre_degree_mat = sp.load_npz(self.path + '/trust_degree.npz')
                norm_adj = pre_degree_mat
            except:
                # trust_degree = []
                # truster = self.truster.tolist()
                # trustee = self.trustee.tolist()
                # for i in tqdm(range(len(truster))):
                #     truster_id = truster[i]
                #     trustee_id = trustee[i]
                #     sim = self.calpcc(self.UserItemRatings[truster_id], self.UserItemRatings[trustee_id],
                #                       truster_id, trustee_id)
                #     sim = sim.tolist()
                #     trust_degree.append(sim[0])
                # self.trust_degree = np.array(trust_degree)
                # self.trustdegree = csr_matrix((self.trust_degree, (self.truster, self.trustee)),
                #                               shape=(self.n_user, self.n_user))
                adj_mat = self.trustdegree                                                  #将信任程度矩阵B乘以对角矩阵
                adj_mat = adj_mat.tolil()
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/trust_degree.npz', norm_adj)
                print("成功保存！！")

            self.trustdegree = self._convert_array_to_csr_matrix(trust_degree)
            if self.split == True:
                self.TrustDegreeB = self._split_A_hat(norm_adj)
            else:
                self.TrustDegreeB = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.TrustDegreeB = self.TrustDegreeB.coalesce().to(world.device)

        return self.TrustDegreeB     # TrustDegreeB等价于卷积公式中的前面一部分。\

    def get_degree_martix_improve(self):
        if self.TrustDegreeB_improve is None:
            try:
                pre_degree_mat = sp.load_npz(self.path + '/trust_degree_improve.npz')
                norm_adj = pre_degree_mat
            except:
                total_degree = 2 * self.trustsize
                avg_degree = total_degree / (2 * self.trust_node_number)
                dmax = math.ceil(np.log(self.trust_node_number) / np.log(avg_degree))
                if dmax > 6:
                    dmax = 6
                dist_martix, predecessors = shortest_path(self.TrustNet, return_predecessors=True, directed=True)
                mask = (dist_martix != np.inf) & (dist_martix > 0) & (dist_martix <= 6)
                dist_martix[np.isinf(dist_martix)] = np.inf
                indirect_matrix = np.multiply(dist_martix, mask.astype(float))
                indirect_matrix = np.where(indirect_matrix == 0, np.nan, indirect_matrix)

                trust_martix = np.divide(np.subtract(dmax + 1, indirect_matrix), dmax)
                trust_martix[np.isnan(trust_martix)] = 0

                dense_rate_martix = self.UserItemRatings.toarray()
                pcc_martix = np.corrcoef(dense_rate_martix)
                pcc_martix = (pcc_martix + pcc_martix.T) / 2
                pcc_martix = pcc_martix * 0.5 + 0.5
                np.fill_diagonal(pcc_martix, 0)
                pcc_martix[np.isnan(pcc_martix)] = 0
                # 推测出来的所有的信任程度矩阵
                fuse_trust_weight = np.divide(np.multiply(2 * pcc_martix, trust_martix),
                                              np.add(pcc_martix, trust_martix))
                fuse_trust_weight[np.isnan(fuse_trust_weight)] = 0
                explicit_weight = np.divide(np.multiply(2 * pcc_martix, self.TrustNet.toarray()),
                                            np.add(pcc_martix, self.TrustNet.toarray()), dtype=np.float32)
                explicit_weight[np.isnan(explicit_weight)] = 0
                implicit_weight = np.multiply(fuse_trust_weight, self.TrustNet.toarray())
                implicit_weight = fuse_trust_weight - implicit_weight
                implicit_mask = (implicit_weight >= 0.6)
                implicit_weight = np.multiply(implicit_weight, implicit_mask, dtype=np.float32)
                fuse_trust_weight = explicit_weight + implicit_weight
                fuse_trust_weight = self._convert_array_to_csr_matrix(fuse_trust_weight)

                adj_mat = fuse_trust_weight
                adj_mat = adj_mat.tolil()
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/trust_degree_improve.npz', norm_adj)

            if self.split == True:
                self.TrustDegreeB_improve = self._split_A_hat(norm_adj)
            else:
                self.TrustDegreeB_improve = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.TrustDegreeB_improve = self.TrustDegreeB_improve.coalesce().to(world.device)

            return self.TrustDegreeB_improve


    def trust_topu(self):
        print("social topu graph construct start")
        print("*********************************")
        truster = self.truster.tolist()
        length = len(truster)
        trustee = self.trustee.tolist()
        trustvalue = []
        temp_degree = self.TrustNet
        alpha = 0.6
        d_inv = np.ones(self.n_user)
        d_eye = sp.diags(d_inv)
        Mat = d_eye.tocsr() - alpha * temp_degree.tocsr().transpose()
        probability = np.empty(shape=[0, self.n_user])
        uni_truster = np.unique(truster)
        uni_truster = uni_truster[::-1]
        dyk = length-1
        for index in uni_truster:
            print(f"第{index}个用户节点开始游走")
            r = np.zeros((self.n_user, 1))
            r[index] = 1-alpha
            res = gmres(Mat, r, tol=1e-8)[0]
            probability = np.append(probability, [res], axis=0)
            while(truster[dyk]==index):
                trustvalue.append(res[trustee[dyk]])
                dyk = dyk-1
                if dyk == length:
                    break
        self.trustvalue = np.array(trustvalue)
        self.trusttopuvalue = csr_matrix((self.trustvalue, (self.truster, self.trustee)),
                                         shape=(self.n_user, self.n_user))
        print("*********************************")
        print("social topu graph completed")
        return self.trusttopuvalue

    def get_degree_martix_topu(self):
        if self.TrustTopuGraph is None:
            try:
                pre_degree_mat = sp.load_npz(self.path + '/trust_degree_topu.npz')
                norm_adj = pre_degree_mat
            except:
                adj_mat = self.trust_topu()
                adj_mat = adj_mat.tolil()
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/trust_degree_topu.npz', norm_adj)

            if self.split == True:
                self.TrustTopuGraph = self._split_A_hat(norm_adj)
            else:
                self.TrustTopuGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.TrustTopuGraph = self.TrustTopuGraph.coalesce().to(world.device)

        return self.TrustTopuGraph

    def get_Fusion_graph(self):
        if self.fusion_graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_Fusion_graph.npz')
                norm_adj = pre_adj_mat
            except:
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                # R = self.UserItemNet.tolil()
                R = self.UserItemRatings.tolil()
                R = R / self.max_score
                # B = self.TrustNet.tolil()
                B = self.trustdegree.tolil()
                adj_mat[:self.n_users, self.n_users:] = R               # 放在右上角
                adj_mat[self.n_users:, :self.n_users] = R.T             # 放在左下角
                adj_mat[:self.n_users, :self.n_users] = B
                # adj_mat = adj_mat.todok()

                norm_adj = adj_mat
                norm_adj = norm_adj.tocoo()
                row = np.array(norm_adj.row.reshape(-1))
                col = np.array(norm_adj.col.reshape(-1))
                dat = np.array(norm_adj.data.reshape(-1))
                sum_all = norm_adj.sum(axis=1).squeeze()
                sum_all = np.array(sum_all).ravel()
                length = len(row)
                for i in range(length):
                    dat[i] = dat[i]/sum_all[row[i]]

                norm_adj = csr_matrix((dat, (row, col)),
                                      shape=(self.n_users + self.m_items, self.n_users + self.m_items), dtype=float)
                norm_adj = norm_adj.tocsr()

                end = time()

                sp.save_npz(self.path + '/s_pre_Fusion_graph.npz', norm_adj)

        return norm_adj

    def pagerank(self):
        print("pagerank start")
        temp_degree = self.trustdegree.todense()      # 用户i对用户j的信任程度为temp_degree[i,j] type:matrix
        temp_degree = np.array(temp_degree)
        for i in range(len(temp_degree[0])):  # 将邻接矩阵转化为转移矩阵
            sum_i = np.sum(temp_degree[i, :])
            if sum_i == 0:
                sum_i = 1
            for j in range(len(temp_degree[0])):
                temp_degree[i, j] = temp_degree[i, j] / sum_i
        r = np.ones((len(temp_degree[0]), 1)) * (1 / len(temp_degree[0]))
        alpha = 0.85
        count = 0  # 记录迭代次数
        d_value = 1000
        while d_value > 0.00000000000001 or count < 1000:
            r_1 = np.dot(temp_degree.T, r)*alpha + (1 - alpha)/len(temp_degree[0])   # 修正的pageRank
            # r_1 = np.dot(temp_degree, r)  # 基本的pageRank
            d_value = r - r_1
            d_value = np.max(np.abs(d_value))
            r = r_1
            count += 1
        r[0] = 0.0
        r = r.squeeze()
        np.save(self.path + '/pagerank_all_user.npy', r)
        top_trust_kuser = sorted(np.argsort(r)[-self.top_k:])
        print(np.array(top_trust_kuser))
        return torch.tensor(np.array(top_trust_kuser))

    def personal_rank_community_1(self):                                # 在用户-物品二分图游走
        sub_str = str(world.sub_num)
        print("personal rank start")
        self.top_trust_kuser = self.pagerank()
        temp_degree = self.get_Fusion_graph()
        alpha = 0.5
        d_inv = np.ones(self.n_user + self.m_item)
        d_eye = sp.diags(d_inv)
        Mat = d_eye.tocsr() - alpha * temp_degree.tocsr().transpose()
        probability = np.empty(shape=[0, self.n_user + self.m_items])
        for index in self.top_trust_kuser:
            print(f"第{index}个中心节点开始游走")
            r = np.zeros((self.n_users + self.m_items, 1))
            r[index] = 1-alpha
            res = gmres(Mat, r, tol=1e-8)[0]
            probability = np.append(probability, [res], axis=0)  # probability shape:self.top_k * self.n_user+self.m_items

        probability_normed = probability / probability.max(axis=0)  # 归一化
        probability_normed[np.isnan(probability_normed)] = 0.0
        probability_normed = probability_normed.T
        probability = torch.tensor(probability_normed)
        probability_observed = np.array(probability)

        probability_ones = torch.ones(self.n_user + self.m_item, self.top_k)
        probability_zeros = torch.zeros(self.n_user + self.m_item, self.top_k)
        one_hot_emb = torch.where(probability >= 0.7, probability_ones, probability_zeros)
        one_hot_emb = np.array(one_hot_emb)
        np.save(self.path + '/centlabel_'+sub_str+'.npy', one_hot_emb)
        # print(probability_observed)
        return one_hot_emb                                      # (self.n_user+self.m_item) * self.top_k


    def get_hypergraph(self):
        sub_str = str(world.sub_num)
        if self.hypergraph is None:
            try:
                norm_adj_1 = sp.load_npz(self.path + '/hyper_graph_1.npz')
                norm_adj_2 = sp.load_npz(self.path + '/hyper_graph_2.npz')
                # norm_adj = sp.load_npz(self.path + '/hyper_graph.npz')
            except:
                adj_mat = self.personal_rank_community_1()
                node_number = np.sum(adj_mat, 0)
                print(node_number)
                adj_mat = self._convert_array_to_csr_matrix(adj_mat)
                adj_mat = adj_mat.tolil()
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1
                v_inv = np.power(rowsum, -0.5).flatten()
                v_inv[np.isinf(v_inv)] = 0.
                v_mat = sp.diags(v_inv)

                colsum = np.array(adj_mat.sum(axis=0))
                colsum[colsum == 0.] = 1
                e_inv = np.power(colsum, -1).flatten()
                e_inv[np.isinf(e_inv)] = 0.
                e_mat = sp.diags(e_inv)

                norm_adj_1 = v_mat.dot(adj_mat)
                norm_adj_1 = norm_adj_1.dot(e_mat)


                norm_adj_2 = adj_mat.T.dot(v_mat)
                norm_adj_1 = norm_adj_1.tocsr()
                norm_adj_2 = norm_adj_2.tocsr()

                # norm_adj = v_mat.dot(adj_mat)
                # norm_adj = norm_adj.dot(e_mat)
                # norm_adj = norm_adj.dot(adj_mat.T)
                # norm_adj = norm_adj.dot(v_mat)
                # norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/hyper_graph_1_'+sub_str+'.npz', norm_adj_1)
                sp.save_npz(self.path + '/hyper_graph_2_'+sub_str+'.npz', norm_adj_2)
                # sp.save_npz(self.path + '/hyper_graph.npz', norm_adj)
        if self.split == True:
            self.hypergraph_1 = self._split_A_hat(norm_adj_1)
            self.hypergraph_2 = self._split_A_hat(norm_adj_2)
            # self.hypergraph = self._split_A_hat(norm_adj)
        else:
            self.hypergraph_1 = self._convert_sp_mat_to_sp_tensor(norm_adj_1)
            self.hypergraph_1 = self.hypergraph_1.coalesce().to(world.device)
            self.hypergraph_2 = self._convert_sp_mat_to_sp_tensor(norm_adj_2)
            self.hypergraph_2 = self.hypergraph_2.coalesce().to(world.device)
            # self.hypergraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            # self.hypergraph = self.hypergraph.coalesce().to(world.device)

        return self.hypergraph_1, self.hypergraph_2
        # return self.hypergraph



    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems