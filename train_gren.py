"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
Euclidean distance

CIA-Net Training code with attention branch using negative sampling based on structural similarity
"""
from __future__ import print_function
import _init_paths
import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
import sys
from model.faster_rcnn.resnet import resnet
import torch.nn.functional as F
from torch.autograd import Variable
#from model.faster_rcnn.seresnet import seresnet

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from torch.utils.data.sampler import Sampler
#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from model.utils.net_utils import save_checkpoint
import cv2
from datasets.imdb import imdb

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Loading arguments
parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      action='store_true',
                      default=False)
parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default='1.0_0.0', type=str)
parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=4, type=int)           
parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=55972, type=int)       

# attention option
parser.add_argument('--attn_type', dest='attn_type',
                      help='attention type: baseline | basic | contextual_vgg_local | contextual_vgg_global | contextual_net_global | contextual_net_local',
                      default="baseline", type=str)

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--s', dest='session',
                      help='training session',
                      default='1.0_0.4', type=str)
parser.add_argument('--fold', default='4', type=str,
                    help='fold to use')
parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=4, type=int)
parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
parser.add_argument('--log_step', default=20, type=int, metavar='N',
                    help='number of step to log loss')
parser.add_argument('--epochs', default=9, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--max_epochs', default=9, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=2, type=int, metavar='N',
                    help='train batchsize (default: 6)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu_id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.imdb_name = "chestXray_train_{}_{}".format(args.fold, args.session)
print(args.imdb_name)
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
   
    self.batch_size = batch_size
    self.range = torch.arange(0, batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

# FeatureExtractor
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            if name in self.extracted_layers: 
                continue
            x = module(x)
        return x

def distance_loss(feature, similar):
    similar = similar
    dis_loss = 0
    feature_set = feature    
    feature_set_list = feature_set.split(1, 0)	
    fea_set = list(feature_set_list)
    for i in range(len(fea_set)):
        for j in range(i+1, len(fea_set)):
            dist2 = F.pairwise_distance(fea_set[i], fea_set[j], p=2) #Euclidean distance
            dis_loss += 0.17 * dist2 
    dis_loss = dis_loss * similar
    return dis_loss

def train(data, model, optimizer, similar):
    similar = similar
    im_data = torch.FloatTensor(1)
    im_data_a = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_classes = torch.FloatTensor(1)
   
    im_data = im_data.cuda()
    im_data_a = im_data_a.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    gt_classes = gt_classes.cuda()
    with torch.no_grad():
         im_data.resize_(data[0].size()).copy_(data[0])  # CPU to GPU
         im_info.resize_(data[1].size()).copy_(data[1])
         gt_boxes.resize_(data[2].size()).copy_(data[2])
         num_boxes.resize_(data[3].size()).copy_(data[3])
         gt_classes.resize_(data[4].size()).copy_(data[4])
    model.zero_grad()
    extract_list = ['CONV_BRIDGE_1','ReLU','CONV_BRIDGE_2','CONV_BRIDGE_ATTN_1','attn1_loc','attn1_attn','attn2_loc','attn2_attn',]#,'RCNN_base',
    features_model = FeatureExtractor(model, extract_list)
    feature = features_model(im_data)
    feature = feature.view(feature.shape[0], -1)
    feature = F.normalize(feature)  
    loss_distance = distance_loss(feature, similar)

    probs, loss_strong, loss_weak, a_map = model(im_data, im_data_a, im_info, gt_boxes, num_boxes, gt_classes,
                                                 attn_type=args.attn_type)
                                      
    loss = loss_strong.mean()*4 + loss_weak.mean() + loss_distance.mean()*0.1 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss_strong.mean(), loss_weak.mean(), loss_distance.mean(), probs.mean(), a_map

def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] *= args.gamma
    for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic='store_true', attn_type=args.attn_type)
    fasterRCNN.create_architecture()
    fasterRCNN.cuda()
    train_size = len(roidb)
    iters_per_epoch = int(train_size / args.train_batch)
    print('{:d} roidb entries'.format(len(roidb)))
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.train_batch,
                           imdb.num_classes, training=True)

    sampler_batch = sampler(train_size, args.train_batch)
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.train_batch,
            sampler=sampler_batch, num_workers=0, pin_memory=False)

    # define loss function (criterion) and optimizer
    optimizer = optim.SGD(fasterRCNN.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Resume
    if args.resume:
      load_name = os.path.join('/home/qbl/CCG_code/checkpoint/g8_4/{}_fold{}_{}_lr{}_decay_epoch_{}/'.format(args.attn_type,
                                                                                        args.fold,
                                                                                        args.session,
                                                                                        args.lr,
                                                                                        args.lr_decay_step)
                               ,'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
      args.start_epoch = args.checkepoch + 1
      print("loading checkpoint %s" % load_name)
      checkpoint = torch.load(load_name)
      fasterRCNN.load_state_dict(checkpoint['model'])
      print("loaded checkpoint %s" % (load_name))

    print("Using Attention: {}".format(args.attn_type))
    save_dir = '/home/qbl/CCG_code/checkpoint/g8_4/{}_fold{}_{}_lr{}_decay_epoch_{}/'.format(args.attn_type,
                                                                                        args.fold,
                                                                                        args.session,
                                                                                        args.lr,
                                                                                        args.lr_decay_step)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("Model saving at {}".format(save_dir))

    # Train
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        data_iter = iter(dataloader)
        out_hash = imdb.hash_image()
    
        import pdb

        fasterRCNN.train()        
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))
        #pdb.set_trace()
        if (epoch+1) % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, epoch)
        for step in range(1, iters_per_epoch + 1):
            data = next(data_iter)
            similar = next(out_hash)
            loss_strong, loss_weak, loss_distance, probs, a_map= train(data, fasterRCNN, optimizer, similar)

            if step % args.log_step==0:
                print("epoch: %.4f, ,step: %.4f, loss: %.4f, lr: %.4f" % (epoch, step, loss_strong*4+loss_weak, state['lr']))
                print("loss_strong: %.4f, loss_weak: %.4f, loss_distance: %.4f, probs: %.4f, a_map: %.4s" % (loss_strong, loss_weak, loss_distance, probs, a_map))
                print(save_dir)

        if (epoch % 1==0):
           save_name = os.path.join(save_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
           save_checkpoint({
           'session': args.session,
           'epoch': epoch + 1,
           'model': fasterRCNN.state_dict(),
           'optimizer': optimizer.state_dict(),
           }, save_name)





if __name__ == '__main__':
    main()





