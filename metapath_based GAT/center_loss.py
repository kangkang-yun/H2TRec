import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
# import world

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, label, user_count, item_count, user_emb, item_emb, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.label = label
        self.label = torch.tensor(self.label).cuda()
        self.user_count = user_count
        self.item_count = item_count
        self.user_emb = user_emb.weight
        self.item_emb = item_emb.weight
        self.centers = nn.Embedding(self.num_classes, self.feat_dim).cuda()
        self.centers = self.centers.weight
        self.constrain = 100

    def forward(self, user_batch, item_batch):
        user_e = F.embedding(user_batch, self.user_emb)
        item_e = F.embedding(item_batch, self.item_emb)
        batch_size = user_e.size(0)
        distmat_user = torch.pow(user_e, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                     torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_item = torch.pow(item_e, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                     torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_user.addmm_(1, -2, user_e, self.centers.t())
        distmat_item.addmm_(1, -2, item_e, self.centers.t())
        mask_user = F.embedding(user_batch, self.label)
        item_batch_temp = item_batch + self.user_count
        mask_item = F.embedding(item_batch_temp, self.label)
        dist_user = distmat_user*mask_user.float()
        dist_item = distmat_item*mask_item.float()
        loss = dist_user.clamp(min=1e-12, max=1e+12).sum()+dist_item.clamp(min=1e-12, max=1e+12).sum()
        size = mask_user.sum().item() + mask_item.sum().item()


        return (loss/size)/self.constrain




