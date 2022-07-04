import numpy as np
import json
from tqdm import tqdm
import cv2
import os


json_path = '/home/feiyu/chestXray/cls_normal_data.json'
data_dir = '/home/feiyu/chestXray/images'
out_dir = '/home/feiyu/chestXray'
hash_size = 16
# data = np.load(os.path.join(out_dir, 'perceptual_hash_lib.npz'))
# import pdb
# pdb.set_trace()
json_file = json.load(open(json_path))
hash_mat = np.zeros((len(json_file.keys()), hash_size * hash_size))
names = []
cnt = 0
for img_path in tqdm(json_file):
    cnt += 1
    img = cv2.imread(os.path.join(data_dir, img_path))
    names.append(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (hash_size, hash_size))
    img_ave = np.mean(img)
    img[img < img_ave] = 0
    img[img > img_ave] = 1
    hash_mat[cnt - 1, :] = img.flatten()
hash_mat = hash_mat.astype(np.bool_)
names = np.array(names)
np.savez(os.path.join(out_dir, 'perceptual_hash_lib_no_affine.npz'), hash_mat=hash_mat, names=names)

