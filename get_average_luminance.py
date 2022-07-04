import cv2
import numpy as np
import os
from tqdm import tqdm
img_dir = '/home/feiyu/chestXray/images_affine'
img_list = os.listdir(img_dir)
img_num = len(img_list)

indexes = np.random.randint(0, img_num, 1000)
maxes = 0
for idx in tqdm(indexes):
    maxes += np.max(cv2.imread(os.path.join(img_dir, img_list[idx])))

print(maxes/1000)
