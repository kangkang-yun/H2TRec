import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import world

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, dataset, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        sub_str = str(world.sub_num)
        self.label = np.load('../data/'+world.dataset+'/centlabel_'+sub_str+'.npy', allow_pickle=True)
        self.label = torch.tensor(self.label).to(world.device)

        if self.use_gpu:
            self.centers = torch.nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.feat_dim).cuda()
        else:
            self.centers = torch.nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.feat_dim)

        nn.init.xavier_uniform_(self.centers.weight, gain=1)
        self.centers = self.centers.weight

    def forward(self, x, batch):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        mask = F.embedding(batch, self.label)
        size = mask.sum().item()
        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum()

        return loss, size
