# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
#import cv2
import torch
from scipy.misc import imread
import scipy.io as sio
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes

from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.densenet import densenet
from model.faster_rcnn.resnext import resnext
from model.utils.other_utils import compute_AUCs, IOU_on_det, IOU
import numpy as np
import pdb
from tqdm import tqdm

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def max1(a, b):
    if a > b:
        return a
    else:
        return b


def min1(a, b):
    if a < b:
        return a
    else:
        return b


def IOU_two_box(pred, gt):
    print(pred)
    x = max1(pred[0], gt[0])
    y = max1(pred[1], gt[1])
    x1 = min1(pred[2], gt[2])
    y1 = min1(pred[3], gt[3])
    intersection = (x1 - x) * (y1 - y)
    if (x1 - x) < 0 or (y1 - y) < 0:
        intersection = 0
    if pred[0] > pred[2] or pred[1] > pred[3]:
        return 0
    union = (pred[2] - pred[0]) * (pred[3] - pred[1]) + (gt[2][0] - gt[0][0]) * (gt[3][0] - gt[1][0]) - intersection
    return float(intersection) / float(union)


def attention_visualization(index, im_ori, img_src, attention, out_dir):
    # pdb.set_trace()
    # attention = attention-np.mean(attention)
    # attention[attention<0] = 0
    # attention = attention/np.max(attention) * 255
    # pdb.set_trace()
    img_att = cv2.cvtColor(attention, cv2.COLOR_GRAY2RGB)
    img_att = cv2.resize(img_att, (256, 256))
    img_src = cv2.resize(img_src, (256, 256))
    im_ori = cv2.resize(im_ori, (256, 256))
    img_att = cv2.applyColorMap(img_att, 2)
    img_att = im_ori * 0.6 + img_att * 0.4
    img_att = img_att.astype(np.uint8)
    out_img = np.zeros((256, 256 * 2, 3))
    out_img[:, :256, :] = img_src
    out_img[:, 256:, :] = img_att
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(str(index))), out_img)


def tensor2np(tensor, flg):
    if flg:
        img = tensor.mul(255).byte()
    else:
        img = tensor + 128
    img = img.cpu().numpy().transpose((1, 2, 0))
    # img = img.numpy().transpose((1, 2, 0))
    return img


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--attn_type', dest='attn_type',
                        help='attention type: baseline | basic | contextual_vgg_local | contextual_vgg_global | contextual_net_global | contextual_net_local',
                        default="baseline", type=str)
    parser.add_argument('--gen_vis', dest='gen_vis',
                        help='generate attention visualization or not',
                        default=False, type=bool)
    parser.add_argument('--do_upsampling', dest='do_upsampling',
                        help='do upsampling in testing or not',
                        default=True, type=bool)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='chestXray', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, resnext50',
                        default='seres50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/home/qbl/attention_models/basic_fold0_0.5_0.8_struct_fast",
                        # default="/home/qbl/pytorch-classification/checkpoints",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',default=True,
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default='1.0_0.4', type=str)# 1.0_0.4
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=2, type=int) #8
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=55972, type=int)#6996  9445  6952  28335 55972
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='directory to write images for demo',
                        default="/data2/qbl/pytorch-classification0/outputs/vis_net_global_ver2")
    parser.add_argument('--gpu_id', default='4', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--fold', default='4', type=str,
                        help='fold to use')
    parser.add_argument('--ver', default='2', type=str,
                        help='ver to use')
    parser.add_argument('--ups_size', default=128, type=str,#########128
                        help='up sampling size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)  #4
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    return args
    #faster_rcnn_1.0_0.4_9_18657.pth


if __name__ == '__main__':
    #baseline_fold4_1.0_0.4_lr0.001_decay_epoch_4
    args = parse_args()#/home/qbl/CCG_code/checkpoint/hash_only/    calc_6   /data2/qbl/pytorch-classification_checkpoint/hash_weight/
    args.load_dir = '/home/qbl/CCG_code/checkpoint/g8_4/{}_fold{}_{}_lr{}_decay_epoch_{}/'.format(args.attn_type,
                                                                                             args.fold,
                                                                                             args.checksession,
                                                                                             args.lr,
                                                                                             args.lr_decay_step)
    # print('Called with args:')
    # print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    hash_data = np.load('/data2/qbl/chest_x_ray/perceptual_hash_lib.npz')
    hash_mat = hash_data['hash_mat']
    hash_names = hash_data['names']
    np.random.seed(cfg.RNG_SEED)
    if args.dataset == 'chestXray':
        args.imdb_name = "dummy"
        if args.checksession == '1.0_0.0':
            args.imdbval_name = "chestXray_test_0_0.0_1.0"
            # args.imdbval_name = "chestXray_test_4_0.0_0.2"
        elif args.checksession == '1.0_0.4':
            args.imdbval_name = "chestXray_test_{}_0.0_0.6".format(args.fold)
        else:
            args.imdbval_name = "chestXray_test_{}_0.0_0.2".format(args.fold)
        # args.imdbval_name = "chestXray_test_4_0.0_0.2"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    attn_imdbval_name = "chestXray_train_0_0.8_0.0"
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    # imdb_a, roidb_a, ratio_list_a, ratio_index_a = combined_roidb(attn_imdbval_name, False)
    imdb.competition_mode(on=True)
    # imdb_a.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    load_name = os.path.join(args.load_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'seres50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic,
                            attn_type=args.attn_type)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'dense121':
        fasterRCNN = densenet(imdb.classes, 121, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'resnext50':
        fasterRCNN = resnext(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)

    for k in list(checkpoint['model'].keys()):
        # print(k)
        if k.find('RCNN_rpn') != -1 or k.find('top') != -1 or k.find('cls_score') != -1 or k.find('bbox') != -1:
            del checkpoint['model'][k]

    fasterRCNN.load_state_dict(checkpoint['model'])
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_data_a = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_classes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_data_a = im_data_a.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_classes = gt_classes.cuda()

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0, pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    # num_images = 5
    pred_all = torch.zeros((num_images, 14), dtype=torch.float32)
    gt_all = torch.zeros((num_images, 14), dtype=torch.uint8)

    chest_classes = ('__background__',  # always index 0
                     'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                     'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                     'Pleural Thickening', 'Pneumonia', 'Pneumothorax')

    total, hits1 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits2 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits3 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits4 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits5 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits6 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    total, hits7 = {i: 0 for i in range(imdb.num_classes)}, {i: 0 for i in range(imdb.num_classes)}
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    up_sampling_size = int(args.ups_size)
    for i in tqdm(range(num_images), unit="image"):
        data = next(data_iter)
        
        im_data.resize_(data[0].size()).copy_(data[0])
        # im_data_a.data.resize_(norm_imgs.size()).copy_(norm_imgs)
        # im_data_a.data.resize_(data_a[0].size()).copy_(ag_a1[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        gt_classes.resize_(data[4].size()).copy_(data[4])
        '''
        # norm_imgs = get_normal_data(data[0], hash_mat, hash_names)
        # ag_a1 = next(ag_a)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        # im_data_a.data.resize_(norm_imgs.size()).copy_(norm_imgs)
        # im_data_a.data.resize_(data_a[0].size()).copy_(ag_a1[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        gt_classes.data.resize_(data[4].size()).copy_(data[4])
        '''
        if int(num_boxes) == 0:  # we only test those have box annotations
            continue
        det_tic = time.time()
        do_upsampling = args.do_upsampling
        # pdb.set_trace()
        probs, loss_strong, loss_weak, attn_return = fasterRCNN(im_data, im_data_a, im_info, gt_boxes, num_boxes,
                                                                gt_classes,
                                                                upsampling=do_upsampling,
                                                                attn_type=args.attn_type,
                                                                up_sampling_size=up_sampling_size)
        # pdb.set_trace()
        fh, fw = probs.size(2), probs.size(3)
        class_names = []
        for n in range(int(num_boxes)):  # for each gt box
            gt_box = gt_boxes[0, n, :-1]
            gt_cls = gt_boxes[0, n, -1]
            class_names.append(chest_classes[int(gt_cls)])
            tmp = probs[:, int(gt_cls) - 1, :, :].squeeze()
            hit = 0
            hit1 = 0
            hit2 = 0
            hit3 = 0
            hit4 = 0
            hit5 = 0
            hit6 = 0
            hit7 = 0

            m = 0

            if int(gt_cls) - 1 in range(20):
                ind = torch.nonzero(tmp > 0.5)
            boxes = torch.zeros((ind.size(0), 4)).cuda()
            for k in range(ind.size(0)):
                y1, x1 = ind[k, 0], ind[k, 1]

                if do_upsampling is True:
                    m1 = (512 / up_sampling_size)
                    m = (512 / up_sampling_size)
                else:
                    m1 = (512 / 16)
                    m = (512 / 16)
                box = torch.tensor((x1 * m, y1 * m1, (x1 + 1) * m, (y1 + 1) * m1))
                boxes[k] = box

            if ind.nelement() > 0:  # have detections
                # pdb.set_trace()
                ov = IOU(boxes, gt_box.unsqueeze(0))
                if ov >= 0.1:
                    hit1 = 1
                if ov >= 0.2:
                    hit2 = 1
                if ov >= 0.3:
                    hit3 = 1
                if ov >= 0.4:
                    hit4 = 1
                if ov >= 0.5:
                    hit5 = 1
                if ov >= 0.6:
                    hit6 = 1
                if ov >= 0.7:
                    hit7 = 1
            if do_upsampling is True:
                m = int(512 / up_sampling_size)
            else:
                m = int(512 / 16)

            if args.gen_vis:
                # src_img = tensor2np(im_data[0], False)
                im_ori = cv2.imread(imdb.image_path_at(i))
                im2show = cv2.resize(im_ori, dsize=(fw * m, fh * m), interpolation=cv2.INTER_CUBIC)
                im2show = vis_detections(im2show, class_names, gt_box.unsqueeze(0).cpu().numpy(), boxes.cpu().numpy(),
                                         0.0)
                index = i
                """
                              if args.attn_type != 'basic':
                    att = attn_return[0][0]
                    att = att - torch.mean(att)
                    att = att / torch.max(att)
                    if args.attn_type == 'contextual_net_global':
                        attn_feat = att.view(1, 1, -1) * 5
                    else:
                        attn_feat = att.view(1, 1, -1) * 1.5
                    p_attn_feat = F.softmax(attn_feat, dim=2)
                    p_attn_feat = p_attn_feat.view(16, 16)
                    p_attn_feat = p_attn_feat / torch.max(p_attn_feat)
                    att = p_attn_feat.mul(255).byte()
                    att = att.cpu().numpy()
                else:
                    att = attn_return[0][0]
                    # att = att - torch.mean(att)
                    # att = att / torch.max(att)
                    # att[att<0]=0
                    att = att.mul(255).byte()
                    att = att.cpu().numpy()
                """
                att = attn_return[0][0]
                att = att.mul(255).byte()
                att = att.cpu().numpy()
                # pdb.set_trace()
                attention_visualization(index, im_ori, im2show, att, args.output_dir)
                # im2show = cv2.imread(imdb.image_path_at(i))
                # im2show = cv2.resize(im2show, dsize=(fw * m, fh * m), interpolation=cv2.INTER_CUBIC)
                # im2show = vis_detections(im2show, class_names, gt_box.unsqueeze(0).cpu().numpy(), boxes.cpu().numpy(), 0.0)
                # result_path = os.path.join(args.output_dir, imdb.image_index[i])
                # print(result_path)
                # cv2.imwrite(result_path, im2show)

            hits1[int(gt_cls)] += hit1
            hits2[int(gt_cls)] += hit2
            hits3[int(gt_cls)] += hit3
            hits4[int(gt_cls)] += hit4
            hits5[int(gt_cls)] += hit5
            hits6[int(gt_cls)] += hit6
            hits7[int(gt_cls)] += hit7
            total[int(gt_cls)] += 1

    valid_cls = [1, 2, 5, 9, 10, 11, 13, 14]
    pred1 = 0
    pred2 = 0
    pred3 = 0
    pred4 = 0
    pred5 = 0
    pred6 = 0
    pred7 = 0
    for i in valid_cls:
        # for i in range(1, imdb.num_classes):
        x = total[i]
        if x == 0:
            x += 1
        if args.checksession == '1.0_0.0':
            print('0.1:', chest_classes[i], hits1[i], total[i], hits1[i] / x)
        # print(chest_classes[i], hits2[i], total[i], hits2[i] / x)
        print('0.3:', chest_classes[i], hits3[i], total[i], hits3[i] / x)
        # print(chest_classes[i], hits4[i], total[i], hits4[i] / x)
        print('0.5:', chest_classes[i], hits5[i], total[i], hits5[i] / x)
        # print(chest_classes[i], hits6[i], total[i], hits6[i] / x)
        print('0.7:', chest_classes[i], hits7[i], total[i], hits7[i] / x)
        pred1 += hits1[i] / x
        pred2 += hits2[i] / x
        pred3 += hits3[i] / x
        pred4 += hits4[i] / x
        pred5 += hits5[i] / x
        pred6 += hits6[i] / x
        pred7 += hits7[i] / x
    print('mean')
    if args.checksession == '1.0_0.0':
        print('0.1:', pred1 / 8)
    # print(pred2 / 8)
    print('0.3:', pred3 / 8)
    # print(pred4 / 8)
    print('0.5:', pred5 / 8)
    # print(pred6 / 8)
    print('0.7:', pred7 / 8)
