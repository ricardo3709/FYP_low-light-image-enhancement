from ast import main
from tokenize import Double
from unittest import result
from unittest.mock import patch
from dataset import RawImageDataset
from unittest.mock import patch
import numpy as np
import math
import torch.nn.functional as F
import os
import rawpy
import cv2
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
from dataset import auto_padding
from itertools import product
import tqdm
from seaborn import heatmap
from model import SeeInDark
from PIL import Image
import time
import glob
import torch.nn as nn
import torch.optim as optim
import imageio
torch.cuda.empty_cache()

input_dir = '/home/wang1423/project/compare/codes/datasets/SID/Sony/short/'
gt_dir = '/home/wang1423/project/compare/codes/datasets/SID/Sony/long/'
result_dir = '/home/wang1423/project/compare/codes/results/GT/'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f"Device: {device}")
device = torch.device('cpu')

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))
# print(test_ids)
# tensor_jpg, tensor_xyz = main_batch_preprocess()
# print("output tensor size: ", tensor_xyz.shape)
test_ids.sort()
# print(test_ids)
for test_id in test_ids:
    gt_path = gt_dir + '%05d_00_10s.ARW' % test_id
    save_path = result_dir + '%05d_00_10s.png' % test_id
    raw = rawpy.imread(gt_path)
    res_rgb = raw.postprocess(
        use_camera_wb=True, use_auto_wb=False, no_auto_bright=True)
    imageio.imsave(str(save_path), res_rgb)
