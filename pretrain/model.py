import numpy as np

import world
import torch
from torch import nn, optim
from time import time

class LightGCN(nn.Module):
    def __init__(self,
                 config,
                 dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_groups = self.dataset.topks
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # self.embedding_group = torch.nn.Embedding(
        #     num_embeddings=self.num_groups, embedding_dim=self.latent_dim)
        self.net_1 = nn.Sequential(
            nn.Linear(self.latent_dim * (self.n_layers + 1), self.latent_dim * 4, bias=True),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Linear(self.latent_dim * 4, self.latent_dim * 3, bias=True),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Linear(self.latent_dim * 3, self.latent_dim * 2, bias=True),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=True),
            nn.LeakyReLU(0.4, inplace=True)
        )
        self.activate = nn.LeakyReLU(0.2, inplace=True)

        self.sigmoid = nn.Sigmoid()
        if self.config['pretrain'] == 0:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            print('use xavier initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # self.TrustDegreeGraph = self.dataset.get_degree_martix_improve()
        self.TrustDegree = self.dataset.get_degree_martix()
        self.Graph = self.dataset.getSparseGraph()
        self.Trust_net = self.dataset.getTrustnetwork()
        # self.TrustTopuGraph = self.dataset.get_degree_martix_topu()
        self.hyper_graph_1, self.hyper_graph_2 = self.dataset.get_hypergraph()
        # self.hyper_graph = self.dataset.get_hypergraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
            # social_graph_droped = self.Trust_net
            social_graph_droped = self.TrustDegree
            # social_graph_droped = self.TrustTopuGraph
            community_graph_droped_1 = self.hyper_graph_1
            community_graph_droped_2 = self.hyper_graph_2
            # community_graph_droped = self.hyper_graph
        all_emb = torch.sparse.mm(g_droped, all_emb)
        embs.append(all_emb)
        for layer in range(self.n_layers-1):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # users, items = torch.split(all_emb, [self.num_users, self.num_items])
                # social_emb = torch.sparse.mm(social_graph_droped, users)
                # all_emb = torch.cat([social_emb, items])
                # community_emb_hyper = torch.mm(community_graph_droped_2, all_emb)
                # community_emb_hyper = torch.mm(community_graph_droped_1, community_emb_hyper)


                # community_emb_hyper = torch.mm(community_graph_droped, all_emb)
                community_emb_hyper = torch.sparse.mm(g_droped, all_emb)
                all_emb = community_emb_hyper
                # all_emb = self.activate(all_emb)
                # users_emb, items_emb = torch.split(community_emb_hyper, [self.num_users, self.num_items])
                # users_emb = torch.cat((social_emb, users_emb), -1)
                # users_emb = self.net_2(users_emb)
                # all_emb = torch.cat([users_emb, items_emb])
            embs.append(all_emb)
        # social_rec = embs[0]
        social_rec = torch.cat((embs[0], embs[1], embs[2], embs[3]), -1)
        # social_rec = torch.cat((embs[0], embs[1], embs[2], embs[3], embs[4]), -1)
        # social_rec = torch.cat((embs[0], embs[1], embs[2]), -1)
        # social_rec = torch.cat((embs[0], embs[1]), -1)
        # social_rec = torch.cat((embs[0], embs[1], embs[2], embs[3], embs[4], embs[5]), -1)
        social_rec = self.net_1(social_rec)

        # social_rec = torch.stack(embs, dim=0)
        # social_rec = social_rec.mean(dim=0)
        users, items = torch.split(social_rec, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def group_e(self):
        return self.embedding_group.weight
