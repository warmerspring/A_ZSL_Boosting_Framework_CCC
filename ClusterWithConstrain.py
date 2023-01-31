#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ClusterWithConstrain.py    
@Contact :
@License :
@Create Time: 2021/11/17 10:46  
@Author: Cui Jun-biao    
@Version: 1.0
@Description
"""
import random
import time

import torch
import torch.nn.functional as F
import torch as t
from torch import nn, Tensor
from torch.nn import Parameter
from torch import optim
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from scipy.optimize import linear_sum_assignment

torch.cuda.manual_seed_all(111)
random.seed(111)
torch.manual_seed(111)


class ClusterWithConstrain(nn.Module):
    Z = 0  # shape=[class_num, sam_num]
    alpha = 0.9

    def __init__(self, sam_num, cluster_num, attribute_num, fea_num,
                 clu_loss_wei=1.0,
                 clu_center_align_loss_wei=0.1,
                 class_label_align_loss_wei=0.1,
                 learning_ratio=1e-3):
        super(ClusterWithConstrain, self).__init__()
        self.cluster_num = cluster_num
        self.sam_num = sam_num
        self.cluster_center_net = ClusterCenterNet(cluster_num=cluster_num, fea_num=attribute_num)
        self.cluster_index_net = ClusterIndexNet(cluster_num=cluster_num, sam_num=sam_num)
        self.AE = Encoder(fea_num=fea_num, att_num=attribute_num).cuda()
        self.clu_loss_wei = clu_loss_wei
        self.clu_center_align_loss_wei = clu_center_align_loss_wei
        self.class_label_align_loss_wei = class_label_align_loss_wei
        self.learning_ratio = learning_ratio

    def __init_Parameters_Optimizer(self, X: Tensor, A: Tensor, Y_hat: Tensor, device=t.device('cpu')):
        """
        :param X: shape=[feature_num, sample_num]
        :param device:
        :return:
        """
        self.AE.eval()
        h = self.AE(X.T)
        X_numpy = h.T.detach().clone()
        XY = torch.mm(X_numpy, Y_hat.T)
        init_cluster_center = torch.diag(1.0 / (torch.sum(Y_hat, 1)))
        init_cluster_center = torch.mm(XY, init_cluster_center).cpu().numpy()
        if X_numpy.is_cuda:
            X_numpy = X_numpy.cpu()
        X_numpy = X_numpy.numpy()
        # k_means = KMeans(n_clusters=self.cluster_num, init="k-means++")
        k_means = KMeans(n_clusters=self.cluster_num, init=init_cluster_center.T)
        k_means.fit(X=X_numpy.transpose())  # {0, 1, 2, 3, ...}
        # k_means = KMeans(n_clusters=self.__cluster_num).fit(X_numpy.transpose())
        cluster_sam_score = np.zeros(shape=[self.cluster_num, k_means.labels_.shape[0]])  # [20, 1155]
        sam_indices = np.array(range(k_means.labels_.shape[0]))  # 1155
        cluster_sam_score[k_means.labels_, sam_indices] = 1.0
        cluster_sam_score = t.tensor(data=cluster_sam_score, dtype=X.dtype, device=X.device)
        cluster_center = t.tensor(data=k_means.cluster_centers_, dtype=X.dtype, device=X.device)
        self.cluster_center_net.init_cluster_center(init_clu_center=cluster_center)
        self.cluster_index_net.init_cluster_index(init_clu_index=cluster_sam_score)

        self.__update_parameters_F(A=A, Y_hat=Y_hat)
        self.__optimizer_clu_center_net = optim.Adam(params=self.cluster_center_net.parameters(),
                                                     lr=self.learning_ratio)
        self.__optimizer_clu_index_net = optim.Adam(params=self.cluster_index_net.parameters(), lr=self.learning_ratio)
        self.__optimizerAE = optim.Adam(params=self.AE.parameters(), lr=self.learning_ratio)
        self.to(device=device)

    def __update_parameters_F(self, A: Tensor, Y_hat: Tensor):
        self.cluster_center_net.eval()
        C = self.cluster_center_net().detach()
        Y = self.cluster_index_net().detach()
        self.__F = t.zeros(size=[self.cluster_num, self.cluster_num])
        cost_matrix = t.zeros([self.cluster_num, self.cluster_num])
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                cost_matrix[i, j] = self.clu_center_align_loss_wei * t.norm(input=A[:, i] - C[:, j],
                                                                            p=2) + self.class_label_align_loss_wei * t.norm(
                    input=Y[i, :] - Y_hat[j, :], p=2)
                # cost_matrix[i, j] = t.norm(input=A[:, i] - C[:, j], p=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix.numpy())
        for i in range(self.cluster_num):
            self.__F[row_ind[i], col_ind[i]] = 1
        self.__F = self.__F.cuda()

    def __update_clu_center_index_net(self, X: Tensor, A: Tensor, Y_hat: Tensor, min_loss_gap=1e-5):
        self.train()
        loss_value = np.inf
        loss_gap = np.inf
        while loss_gap > min_loss_gap:
            self.__optimizer_clu_center_net.zero_grad()
            self.__optimizer_clu_index_net.zero_grad()
            self.__optimizerAE.zero_grad()
            loss = self.__loss_function(X=X, A=A, Y_hat=Y_hat)
            loss.backward()
            self.__optimizer_clu_center_net.step()
            self.__optimizer_clu_index_net.step()
            self.__optimizerAE.step()

            new_loss_value = loss.detach().clone()
            if new_loss_value.is_cuda:
                new_loss_value = new_loss_value.cpu()
            new_loss_value = new_loss_value.numpy()
            loss_gap = np.abs(new_loss_value - loss_value)
            loss_value = new_loss_value

        return loss_value

    def __loss_function(self, X: Tensor, Y_hat: Tensor, A: Tensor) -> Tensor:
        """

        :param X: shape=[feature_num, sample_num]
        :param Y_hat: shape=[class_num, sample_num]
        :param A: shape=[feature_num, class_num]
        :return:
        """
        h = self.AE(X.T)  # h=[1155,1024]

        C = self.cluster_center_net()  # [1024, 20]
        Y = self.cluster_index_net()  # [20,1155]# softmax on it
        diff = h.T - t.mm(input=C, mat2=Y)
        diff = t.norm(input=diff, p='fro')
        loss1 = diff * diff / h.numel()
        loss1 = self.clu_loss_wei * loss1

        diff = C - t.mm(input=A, mat2=self.__F)
        diff = t.norm(input=diff, p='fro')
        loss2 = diff * diff / C.numel()
        loss2 = self.clu_center_align_loss_wei * loss2

        diff = Y - t.mm(input=self.__F.T, mat2=Y_hat)
        diff = t.norm(input=diff, p='fro')
        loss3 = diff * diff / Y_hat.numel()
        loss3 = self.class_label_align_loss_wei * loss3

        loss = loss1 + loss2 + loss3
        return loss

    def train_net(self, X: Tensor, A: Tensor, Y_hat: Tensor,
                  max_ite_num=100, test_unseen_label=None, device=t.device('cpu')):
        self.__init_Parameters_Optimizer(X=X, A=A, Y_hat=Y_hat, device=device)
        ite_num = 0
        while ite_num < max_ite_num:
            loss_value = self.__update_clu_center_index_net(X=X, A=A, Y_hat=Y_hat)
            self.__update_parameters_F(A=A, Y_hat=Y_hat)
            ite_num += 1
            pre_Y = self.__get_predict_Y()
            pre_label = np.argmax(pre_Y.T.cpu().detach().numpy(), 1)
            nmi = NMI(test_unseen_label.cpu(), pre_label)
            if ite_num % 2 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                      + ' before ' + str(ite_num) + ' iteration'
                      + ' loss=' + str(loss_value)
                      + ' nmi=' + str(nmi))

    def __get_predict_Y(self) -> Tensor:
        """

        :return:  shape=[class_num, sample_num]
        """
        self.cluster_index_net.eval()
        Y = self.cluster_index_net()
        pre_Y = t.mm(input=self.__F, mat2=Y)
        return pre_Y

    def get_high_confidence_predict_Y(self, sam_num_per_class=10):
        """

        :return:  shape=[class_num, sample_num]
        """
        self.cluster_index_net.eval()
        with torch.no_grad():
            Y = self.cluster_index_net()
            pre_Y = t.mm(input=self.__F, mat2=Y)
            pre_Y = pre_Y.detach()
        ClusterWithConstrain.Z = ClusterWithConstrain.alpha * pre_Y + \
                                 (1 - ClusterWithConstrain.alpha) * ClusterWithConstrain.Z

        pre_label = t.argmax(input=ClusterWithConstrain.Z.T, dim=1)
        pre_label = pre_label.cpu().numpy()  # shape=[sam_num]

        ZZ = ClusterWithConstrain.Z.T
        logZZ = t.log2(input=ZZ)
        ent = - t.sum(input=ZZ * logZZ, dim=1).cpu().numpy()  # shape=[sam_num]

        all_sam_ind = np.array(range(self.sam_num))  # shape=[sam_num]
        select_sam_indices = np.empty(shape=[0], dtype=np.int64)
        select_sam_label = np.empty(shape=[0], dtype=np.int64)
        for class_label in range(self.cluster_num):
            class_sam_ind = all_sam_ind[pre_label == class_label]
            class_sam_ent = ent[class_sam_ind]
            sort_index = np.argsort(a=class_sam_ent, order=None)  # asc
            select_sam_num = sam_num_per_class
            if sam_num_per_class > class_sam_ind.shape[0]:
                select_sam_num = class_sam_ind.shape[0]
            select_sort_index = sort_index[0:select_sam_num]
            select_class_sam_indices = class_sam_ind[select_sort_index]
            select_sam_indices = np.concatenate((select_sam_indices, select_class_sam_indices),
                                                axis=0)
            select_sam_label = np.concatenate((select_sam_label,
                                               np.array([class_label] * select_sam_num)),
                                              axis=0)
        select_sam_indices = torch.from_numpy(select_sam_indices).long()
        select_sam_label = torch.from_numpy(select_sam_label).long()
        return select_sam_indices, select_sam_label


class ClusterCenterNet(nn.Module):
    def __init__(self, cluster_num, fea_num):
        super(ClusterCenterNet, self).__init__()
        self.cluster_num = cluster_num
        self.fea_num = fea_num
        self.net = nn.Linear(in_features=fea_num, out_features=cluster_num, bias=False)

    def init_cluster_center(self, init_clu_center):
        self.net.weight = nn.Parameter(init_clu_center)

    def forward(self):
        E = t.eye(n=self.fea_num).cuda()
        return self.net(input=E)


class ClusterIndexNet(nn.Module):
    def __init__(self, cluster_num, sam_num):
        super(ClusterIndexNet, self).__init__()
        self.cluster_num = cluster_num
        self.sam_num = sam_num
        self.net = nn.Linear(in_features=cluster_num, out_features=sam_num, bias=False)

    def init_cluster_index(self, init_clu_index):
        self.net.weight = nn.Parameter(init_clu_index.T)

    def forward(self):
        E = t.eye(n=self.cluster_num).cuda()
        return F.softmax(input=self.net(input=E), dim=0)


class Encoder(nn.Module):

    def __init__(self, fea_num, att_num):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=fea_num, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=att_num)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
