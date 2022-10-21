from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import glob
import os
import torch
import math
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
import pandas as pd

recon_dir = '/home/wang1423/project/compare/codes/results/reconstructed_results/'
compress_refer_dir = '/home/wang1423/project/compare/codes/results/compress_refer_results/'
ori_raw_dir = '/home/wang1423/project/compare/codes/results/originalRAW_results/'
gt_dir = '/home/wang1423/project/compare/codes/results/GT/'

# get test IDs
test_fns = glob.glob(gt_dir + '*.png')
test_ids = []

for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

test_ids.sort()
exposure_list = [100, 250, 300]
dir_list = [recon_dir, compress_refer_dir, ori_raw_dir]
result_dataframe = pd.DataFrame()

for input_dir in dir_list:
    mse_list = []
    psnr_list = []
    img_name_list = []
    current_batch = input_dir.split('/', -1)[-2]
    current_batch = current_batch.split('_', -1)[0]
    for test_id in test_ids:
        original_img_path = gt_dir + '%05d_00_10s.png' % test_id
        for exposure in exposure_list:
            file_name = '%05d_00_' % test_id + '%03d_out.png' % exposure
            img_name_list.append(file_name)
            reconstructed_img_path = input_dir + str(file_name)
            original_img = Image.open(original_img_path)
            reconstructed_img = Image.open(reconstructed_img_path)
            ori = to_tensor(original_img)
            recon = to_tensor(reconstructed_img)

            # MSE = mean_squared_error(original_img, reconstructed_img)
            # PSNR = peak_signal_noise_ratio(original_img, reconstructed_img)
            mse = torch.mean((ori - recon)**2).item()
            mse_list.append(mse)
            psnr = -10 * math.log10(mse)
            psnr_list.append(psnr)

            # f.write('image_name:' + str(file_name) + '\n')
            # f.write('MSE: ' + str(mse) + '\n')
            # f.write('PSNR: ' + str(psnr) + '\n')
            # f.write('\n')

            # print(test_id)
            # print('MSE:', mse)
            # print('PSNR:', psnr)
    mse_name = str(current_batch + ':MSE')
    psnr_name = str(current_batch + ':PSNR')
    result_dataframe[mse_name] = mse_list
    result_dataframe[psnr_name] = psnr_list

result_dataframe.insert(0, 'image_name', img_name_list)
result_dataframe.to_csv(
    "/home/wang1423/project/compare/codes/results/psnr_result.csv", index=False, sep=',')
