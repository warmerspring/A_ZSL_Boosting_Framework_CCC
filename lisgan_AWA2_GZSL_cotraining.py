from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util
import classifier
import classifier2
import sys
import model
import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ClusterWithConstrain import ClusterWithConstrain

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='AWA2')
parser.add_argument('--dataroot', default='/home/yq/PycharmProjects/xlsa17/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=1800, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='MLP_G')
parser.add_argument('--netD_name', default='MLP_CRITIC')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints', default='AWA2')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=9182, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--ratio', type=float, default=0.1, help='ratio of easy samples')
parser.add_argument('--proto_param1', type=float, default=1e-3, help='proto param 1')
parser.add_argument('--proto_param2', type=float, default=3e-5, help='proto param 2')
parser.add_argument('--loss_syn_num', type=int, default=20, help='number of real clusters')
parser.add_argument('--n_clusters', type=int, default=3, help='number of real clusters')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # -1表示不使用GPU，0表示使用GPU


def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


print(GetNowTime())
print('Begin run!!!')
since = time.time()

opt = parser.parse_args()
print(
    'Params: dataset={:s}, GZSL={:s}, ratio={:.1f}, cls_weight={:.4f}, proto_param1={:.4f}, proto_param2={:.4f}'.format(
        opt.dataset, str(opt.gzsl), opt.ratio, opt.cls_weight, opt.proto_param1, opt.proto_param2))
sys.stdout.flush()

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# device = torch.device('cpu')
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    device = torch.device('cuda:0')
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)

# initialize generator and discriminator
netG = torch.load(f='/home/yq/PycharmProjects/LisGAN-master/AWA2/netG')
# netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)

# netD = model.MLP_CRITIC(opt)

netD = torch.load(f='/home/yq/PycharmProjects/LisGAN-master/AWA2/netD')

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def next_seen_unseen_batch(seen_batch, train_feature, train_label, attribute):
    batch_sam_indices = torch.randperm(train_feature.shape[0])[0:seen_batch]
    batch_feature = torch.from_numpy(train_feature[batch_sam_indices]).float()
    batch_label = torch.from_numpy(train_label[batch_sam_indices]).long()
    batch_att = attribute[batch_label]
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    input_label.copy_(util.map_label(batch_label, data.allclasses))
    opt.batch_sam_indices = batch_sam_indices


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


def generate_syn_feature_with_grad(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_att))
    return syn_feature, syn_label.cpu()


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train_feature = data.train_feature
# train_label = data.train_label.numpy()
#####write the result into the file####
time_str = time.strftime('%Y-%m-%d=%H-%M-%S', time.localtime())
os.makedirs(name='/E/yq/co_training/AWA2/' + time_str + '/')
result_dir = '/E/yq/co_training/AWA2/' + time_str + '/'
result_file = open(file=result_dir + 'LisGAN_cotraining_clustering.txt', mode='w')
##########################################
test_seen_feature = data.test_seen_feature
test_unseen_feature = data.test_unseen_feature
if opt.gzsl:
    test_unlabeled_feature = torch.cat((test_seen_feature, test_unseen_feature), dim=0)
    test_unlabeled_label = torch.cat((data.test_seen_label, data.test_unseen_label), dim=0)
    attribute_matrix = data.attribute
    cluster_num = data.attribute.shape[0]
else:
    test_unlabeled_feature = test_unseen_feature
    test_unlabeled_label = data.test_unseen_label
    attribute_matrix = data.attribute[torch.unique(data.test_unseen_label)]
    cluster_num = data.unseenclasses.shape[0]  # 20
syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
train_X = torch.cat((data.train_feature, syn_feature), 0)
train_Y = torch.cat((data.train_label, syn_label), 0)
nclass = opt.nclass_all
cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 50,
                             2 * opt.syn_num, True)
print('initiation acc:',
      'seen_fc_acc:', cls.acc_fc_seen.item(), 'unseen_fc_acc:', cls.acc_fc_unseen.item(), 'h_fc_acc:',
      cls.acc_fc_H.item(),
      'seen_first_acc:', cls.acc_first_seen.item(), 'unseen_first_acc', cls.acc_first_unseen.item(), 'h_first_acc:',
      cls.acc_first_H.item(),
      file=result_file)
result_file.flush()
cls.best_classifier.eval()
Y_hat = cls.best_classifier(test_unlabeled_feature.cuda())
pre_label1 = np.argmax(Y_hat.detach().cpu().numpy(), 1)
nmi = NMI(test_unlabeled_label, pre_label1)
print(nmi)
#############0~1##############
zsl_Y = torch.exp(Y_hat.detach())
##############0 or 1##############
##############0 or 1##############
# zsl_Y = torch.zeros(Y_hat.shape)
# for index_i in range(Y_hat.shape[0]):
#     zsl_Y[index_i, pre_label1[index_i]] = 1
# zsl_Y = zsl_Y.cuda()
#################################
iii = 0
# class_label_align_loss_wei_arr = [0.1, 0.2, 0.4, 0.8, 0.8, 0.8]
# class_label_align_loss_wei_arr = [0.05, 0.1, 0.2, 0.4, 0.6]
class_label_align_loss_wei_arr = [0.5, 0.5, 0.6, 0.6, 0.8, 0.8]
for epoch_total in range(10):
    ###########clustering#############
    sample_num = test_unlabeled_feature.shape[0]  # 1155
    feature_num = test_unlabeled_feature.shape[1]  # 2048
    attribute_num = data.attribute.shape[1]  # 1024
    if epoch_total % 2 == 0:
        class_label_align_loss_wei = class_label_align_loss_wei_arr[iii]
        iii = iii + 1
    obj = ClusterWithConstrain(sam_num=sample_num, cluster_num=cluster_num, attribute_num=attribute_num,
                               fea_num=feature_num,
                               clu_loss_wei=1.0,
                               clu_center_align_loss_wei=0.1,
                               class_label_align_loss_wei=class_label_align_loss_wei,
                               learning_ratio=0.01)
    obj.train_net(X=test_unlabeled_feature.T.cuda(), A=attribute_matrix.T.cuda(), Y_hat=zsl_Y.T,
                  max_ite_num=100,
                  test_unseen_label=test_unlabeled_label,
                  device=torch.device('cuda:0'))
    # if epoch_total < 3:
    #     sam_num_per_class = int(np.floor(sample_num * (epoch_total + 1) / 3 /50))
    # else:
    #     sam_num_per_class = int(np.floor(sample_num * 2 / 3 / 50))
    sam_num_per_class = int(np.floor(sample_num * (epoch_total + 1) / 3 / 50))
    select_sam_indices, select_sam_label = obj.get_high_confidence_predict_Y(sam_num_per_class=sam_num_per_class)
    if opt.gzsl:
        pre_label = data.allclasses[select_sam_label]
    else:
        pre_label = data.unseenclasses[select_sam_label]
    nmi = NMI(test_unlabeled_label[select_sam_indices], pre_label)
    print(nmi)

    train_feature = np.concatenate((data.train_feature, test_unlabeled_feature[select_sam_indices, :]), axis=0)
    train_label = np.concatenate((data.train_label, pre_label), axis=0)

    # train a classifier on seen classes, obtain \theta of Equation (4)
    nclass = opt.nclass_all
    pretrain_cls = classifier.CLASSIFIER(torch.from_numpy(train_feature), torch.from_numpy(train_label),
                                         nclass, opt.resSize, opt.cuda, 0.001, 0.5, 100, 100,
                                         opt.pretrain_classifier)

    # freeze the classifier during the optimization
    for p in pretrain_cls.model.parameters():  # set requires_grad to False
        p.requires_grad = False
    cls = None
    for epoch in range(opt.nepoch):
        FP = 0
        mean_lossD = 0
        mean_lossG = 0

        for i in range(0, data.ntrain, opt.batch_size):

            for p in netD.parameters():
                p.requires_grad = True

            for iter_d in range(opt.critic_iter):
                next_seen_unseen_batch(opt.batch_size, train_feature, train_label, data.attribute)
                netD.zero_grad()
                sparse_real = opt.resSize - input_res[1].gt(0).sum()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, input_attv)
                fake_norm = fake.data[0].norm()
                sparse_fake = fake.data[0].eq(0).sum()
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one)

                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            netG.zero_grad()
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            criticG_fake = netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            # classification loss
            c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

            labels = Variable(input_label.view(opt.batch_size, 1))
            real_proto = Variable(data.real_proto.cuda())
            dists1 = pairwise_distances(fake, real_proto)
            min_idx1 = torch.zeros(opt.batch_size, data.allclasses.shape[0])
            for i in range(data.train_cls_num):
                min_idx1[:, i] = torch.min(dists1.data[:, i * opt.n_clusters:(i + 1) * opt.n_clusters], dim=1)[
                                     1] + i * opt.n_clusters
            min_idx1 = Variable(min_idx1.long().cuda())
            loss2 = dists1.gather(1, min_idx1).gather(1, labels).squeeze().view(-1).mean()

            seen_feature, seen_label = generate_syn_feature_with_grad(netG, data.allclasses, data.attribute,
                                                                      opt.loss_syn_num)
            # syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            seen_mapped_label = map_label(seen_label, data.seenclasses)
            transform_matrix = torch.zeros(data.train_cls_num, seen_feature.size(0))  # 150x7057
            for i in range(data.train_cls_num):
                sample_idx = (seen_mapped_label == i).nonzero().squeeze()
                if sample_idx.numel() == 0:
                    continue
                else:
                    cls_fea_num = sample_idx.numel()
                    transform_matrix[i][sample_idx] = 1 / cls_fea_num * torch.ones(1, cls_fea_num).squeeze()
            transform_matrix = Variable(transform_matrix.cuda())
            fake_proto = torch.mm(transform_matrix, seen_feature)  # 150x2048
            dists2 = pairwise_distances(fake_proto, Variable(data.real_proto.cuda()))  # 150 x 450
            min_idx2 = torch.zeros(data.train_cls_num, data.train_cls_num)
            for i in range(data.train_cls_num):
                min_idx2[:, i] = torch.min(dists2.data[:, i * opt.n_clusters:(i + 1) * opt.n_clusters], dim=1)[
                                     1] + i * opt.n_clusters
            min_idx2 = Variable(min_idx2.long().cuda())
            lbl_idx = Variable(torch.LongTensor(list(range(data.train_cls_num))).cuda())
            loss1 = dists2.gather(1, min_idx2).gather(1, lbl_idx.unsqueeze(1)).squeeze().mean()

            errG = G_cost + opt.cls_weight * c_errG + opt.proto_param2 * loss2 + opt.proto_param1 * loss1
            errG.backward()
            optimizerG.step()

        print('EP[%d/%d]************************************************************************************' % (
            epoch, opt.nepoch))

        # evaluate the model, set G to evaluation mode
        netG.eval()
        # Generalized zero-shot learning
        if opt.gzsl:
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            nclass = opt.nclass_all
            cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 50,
                                         2 * opt.syn_num, True)
            ####save the best classifier model
            torch.save(cls.best_classifier, f='/home/yq/PycharmProjects/LisGAN-master/AWA2/classifier')
            # print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        # Zero-shot learning
        else:
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                         data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50,
                                         2 * opt.syn_num,
                                         False, opt.ratio, epoch)
            # acc = cls.acc
            # print('unseen class accuracy= ', cls.acc)
        # del cls
        # cls = None
        # # reset G to training mode
        netG.train()
        sys.stdout.flush()
    print('epoch_total:', epoch_total,
          'seen_fc_acc:', cls.acc_fc_seen.item(), 'unseen_fc_acc:', cls.acc_fc_unseen.item(), 'h_fc_acc:',
          cls.acc_fc_H.item(),
          'seen_first_acc:', cls.acc_first_seen.item(), 'unseen_first_acc', cls.acc_first_unseen.item(), 'h_first_acc:',
          cls.acc_first_H.item(),
          file=result_file)
    result_file.flush()
    best_classifier = torch.load(f='/home/yq/PycharmProjects/LisGAN-master/AWA2/classifier')
    Y_hat = best_classifier(test_unlabeled_feature.cuda())
    pre_label1 = np.argmax(Y_hat.detach().cpu().numpy(), 1)
    nmi = NMI(test_unlabeled_label, pre_label1)
    print(nmi)
    #############0~1##############
    zsl_Y = torch.exp(Y_hat.detach())
    ##############0 or 1##############
    # zsl_Y = torch.zeros(Y_hat.shape)
    # for index_i in range(Y_hat.shape[0]):
    #     zsl_Y[index_i, pre_label1[index_i]] = 1
    # zsl_Y = zsl_Y.cuda()
    #################################
time_elapsed = time.time() - since
print('End run!!!')
print('Time Elapsed: {}'.format(time_elapsed))
print(GetNowTime())
