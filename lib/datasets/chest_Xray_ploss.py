from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
import os.path as osp
import json
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
import pdb

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

class chest_Xray(imdb):
    def __init__(self, image_set, fold, cls_ratio, box_ratio):
        imdb.__init__(self, 'chestXray_' + image_set + '_' + fold + '_' + cls_ratio + '_' + box_ratio)
        self._image_set = image_set
        self._fold = fold
        self._cls_ratio = cls_ratio
        if cls_ratio == 0:
            self._cls_ratio = 1
        self._box_ratio = box_ratio
        self._data_path = '../faster-RCNN/data/chestXray'

        self._classes = ('__background__',  # always index 0
                         'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                        'Pleural Thickening', 'Pneumonia', 'Pneumothorax')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert osp.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images_affine_perceptual', index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = osp.join(self._data_path, 'image_lists', self._image_set + '_' +
                                  self._cls_ratio + '_' + self._box_ratio + '_list_fold' + self._fold + '.txt')
        assert osp.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split()[0] for x in f.readlines()]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        cls_anno, box_anno = {}, {}
        if self._cls_ratio != '0.0':
            cls_anno_file = osp.join(self._data_path, 'cls_annotations_' + self._cls_ratio + '_affine_perceptual.json')
            assert osp.exists(cls_anno_file), 'Path does not exist: {}'.format(cls_anno_file)
            cls_anno = json.load(open(cls_anno_file))

        if self._box_ratio != '0.0':
            box_anno_file = osp.join(self._data_path, 'box_annotations_' + self._box_ratio +
                                     '_fold' + self._fold + '_affine_perceptual.json')
            assert osp.exists(box_anno_file), 'Path does not exist: {}'.format(box_anno_file)
            box_anno = json.load(open(box_anno_file))
        if self._cls_ratio == '0.0' and self._box_ratio == '0.0':
            cls_anno_file = osp.join(self._data_path, 'cls_normal_data.json')
            box_anno_file = osp.join(self._data_path, 'box_normal_data.json')
            cls_anno = json.load(open(cls_anno_file))
            box_anno = json.load(open(box_anno_file))


        gt_roidb = [self._load_annotation(index, cls_anno, box_anno) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self, index, cls_anno, box_anno):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # pdb.set_trace()
        if index in box_anno:
            num_objs = len(box_anno[index]['cls'])
        elif index in cls_anno:
            num_objs = len(cls_anno[index])
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        has_boxes = np.zeros((num_objs), dtype=np.int32)

        if index in box_anno:
            boxes_dict = box_anno[index]  # e.g. {'boxes': [[339.16, 119.19, 172.29, 351.08]], 'cls': [9]}
            box_list = boxes_dict['boxes']
            cls_list = boxes_dict['cls']
            for ix, bbox in enumerate(box_list):
                x1 = bbox[0] - 1
                y1 = bbox[1] - 1
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                cls = cls_list[ix]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                has_boxes[ix] = 1
        elif index in cls_anno:
            for ix, cls in enumerate(cls_anno[index]):
                gt_classes[ix] = cls
                has_boxes[ix] = 0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'has_boxes': has_boxes}


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
