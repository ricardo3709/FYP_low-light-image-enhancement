from ast import main
from ctypes.wintypes import RGB
from tokenize import Double
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
m_path = '/home/wang1423/project/compare/codes/Sony_test/saved_model/'
m_name = 'checkpoint_sony_e4000.pth'
result_dir = '/home/wang1423/project/compare/codes/results/reconstructed_results'
compress_result_dir = '/home/wang1423/project/compare/codes/results/compress_refer_results'

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


def RGB_to_Bayer(RGB_tensor):
    tensor_shape = RGB_tensor.shape  # [1,3,H,W]
    H = tensor_shape[2]
    W = tensor_shape[3]
    RGB_tensor = RGB_tensor[:, :, 48:H-48:1, 48:W-48:1]
    tensor_shape = RGB_tensor.shape  # [1,3,H,W]
    H = tensor_shape[2]
    W = tensor_shape[3]
    R_layer = RGB_tensor[:, 0, 0:H:2, 0:W:2].unsqueeze(1)
    G_layer = RGB_tensor[:, 1, 0:H:2, 0:W:2].unsqueeze(1)
    B_layer = RGB_tensor[:, 2, 0:H:2, 0:W:2].unsqueeze(1)
    # R_tensor = torch.from_numpy(R_layer).unsqueeze(1)
    # G_tensor = torch.from_numpy(G_layer).unsqueeze(1)
    # B_tensor = torch.from_numpy(B_layer).unsqueeze(1)
    # out = np.concatenate((R_tensor, G_tensor, B_layer, G_layer), axis=1)
    out_tensor = torch.concat([R_layer, G_layer, B_layer, G_layer], dim=1)
    print('out_tensor shape : ', out_tensor.shape)
    # out_tensor = torch.from_numpy(out).unsqueeze(0)
    return out_tensor


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = np.maximum(raw - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    print("___RAW___")
    print("shape bef", img_shape)

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    print("shape aft", out.shape)
    return out


def pack_jpg(jpg):
    # pack jpg image to 4 channels
    im = np.maximum(jpg - 512, 0) / (16383 - 512)  # subtract the black level

    # im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    print("___JPG___")
    print("shape bef", img_shape)

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=1)  # Need to find a way to 1.sample 2.change dimension to 4(or 3?)
    print("shape aft", out.shape)
    return out


def main_batch_preprocess():
    dataset = RawImageDataset(
        "./datasets/SID/Sony/short/", split="train")
    #dataset = RawImageDataset("./datasets/fivek_dataset", split="val")
    # for i in range(5):
    psnr_list = []
    # Use only first image
    _, raw_file, raw_path = dataset[1]
    #

    res_rgb = raw_file.postprocess(
        use_camera_wb=True, use_auto_wb=False, no_auto_bright=True)
    # image = Image.fromarray(rgb_jpeg)
    image = to_pil_image(res_rgb)
    if False:
        output = BytesIO()
        image.save(output, 'JPEG', quality=90)
        # image.save("test_out.jpg", 'JPEG', quality=90)
        output.seek(0)
        image = Image.open(output)

    tensor_jpg = to_tensor(image)

    # rgb_from_xyz = cv2.cvtColor(res, cv2.COLOR_XYZ2RGB)
    # to_pil_image(rgb_from_xyz)

    # Define xyz tensor
    res_xyz = raw_file.postprocess(output_color=rawpy.ColorSpace(
        5), use_camera_wb=True, use_auto_wb=False, no_auto_bright=True, output_bps=16)  # 0 Raw 5 XYZ
    tensor_xyz = to_tensor(res_xyz/65535).float()

    # rec_xyz, tensor_xyz = reconstrcut(
    #     tensor_jpg, tensor_xyz, desired_ratio=0.0002)
    # psnr = compute_psnr(rec_xyz, tensor_xyz)
    # print("PSNR: %.4f" % psnr)
    # psnr_list.append(psnr)
    # print(np.mean(psnr_list))
    print("jpg tensor:", tensor_jpg.shape)
    print("xyz tensor:", tensor_xyz.shape)
    return tensor_jpg, tensor_xyz

    # # edit part
    # res_xyz = raw_file.raw_image_visible.astype(np.float32)
    # res_xyz = np.expand_dims(res_xyz, axis=2)
    # res_xyz_shape = res_xyz.shape
    # H = res_xyz_shape[0]
    # W = res_xyz_shape[1]

    # xyz_out = np.concatenate((res_xyz[0:H:2, 0:W:2, :],
    #                           res_xyz[0:H:2, 1:W:2, :],
    #                           res_xyz[1:H:2, 1:W:2, :],
    #                           res_xyz[1:H:2, 0:W:2, :]), axis=2)
    # xyz_out = np.expand_dims(xyz_out, axis=0)
    # xyz_full = np.minimum(xyz_out, 1.0)
    # tensor_xyz = torch.from_numpy(
    #     xyz_full).permute(0, 3, 1, 2).to(device)

    # # tensor_xyz = to_tensor(res_xyz/65535).float()
    # tensor_jpg = to_tensor(res_rgb)
    # return tensor_jpg, tensor_xyz


def reconstruct_preprocess(img_index, mode, test_id):
    model = SeeInDark()
    model.load_state_dict(torch.load(
        m_path + m_name, map_location=torch.device('cpu')))
    model = model.to(device)

    dataset = RawImageDataset(
        "./datasets/SID/Sony/short/", split="train")
    #dataset = RawImageDataset("./datasets/fivek_dataset", split="val")
    # for i in range(5):
    psnr_list = []
    # Use only first image
    _, raw_file, raw_path = dataset[img_index]
    in_path = raw_path
    _, in_fn = os.path.split(in_path)
    # print(in_fn)
    gt_path = gt_dir + '%05d_00_10s.ARW' % test_id
    _, gt_fn = os.path.split(gt_path)
    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure/in_exposure, 300)
    # print(ratio)
    ###
    raw = rawpy.imread(raw_path)
    ###

    im = raw.raw_image_visible.astype(np.float32)
    # input_full = np.expand_dims(pack_raw(im), axis=0) * ratio
    # input_full = np.minimum(input_full, 1.0)

    # tensor_xyz = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)
    res_xyz = raw.postprocess(output_color=rawpy.ColorSpace(
        5), use_camera_wb=True, use_auto_wb=False, no_auto_bright=True, output_bps=16)  # 0 Raw 5 XYZ
    tensor_xyz = to_tensor(res_xyz/65535).float()

    if mode == 'lossless':
        # for using lossless image as reference
        print("current recon mode: lossless")
        res_rgb = raw.postprocess(output_color=rawpy.ColorSpace(
            5), use_camera_wb=True, use_auto_wb=False, no_auto_bright=True)
        image = to_pil_image(res_rgb)
        tensor_jpg = to_tensor(image)
    elif mode == 'compress':
        # for using compressed image as reference:
        print("current recon mode: compress")
        res_rgb = raw.postprocess(output_color=rawpy.ColorSpace(
            5), gamma=(1, 1), output_bps=16,
            use_camera_wb=True, use_auto_wb=False, no_auto_bright=True)
        imageio.imsave('temp_reference.jpg', res_rgb)
        image = Image.open('temp_reference.jpg')
        tensor_jpg = to_tensor(image)
    else:
        print("invalid mode")

    return tensor_jpg, tensor_xyz, raw_path


def reconstrcut(jpeg, linear_space_input, patch_size=128, neighbourhood_size=512, desired_ratio=0.002):
    """_summary_
    Args:
        jpeg (torch.Tensor): the image after ISP with the shape 3*H*W
        linear_space_input (torch.Tensor): the linear space image, e.g., RAW space and XYZ space, with the shape 3*H*W
        patch_size (int, optional): _description_. Defaults to 100.
        neighbourhood_size (int, optional): _description_. Defaults to 500.
        desired_ratio (float, optional): _description_. Defaults to 0.002.
    Returns:
        _type_: _description_
    """
    sampling_ratio = round(math.sqrt(1/desired_ratio))
    print("sampling ratio: ", sampling_ratio)
    tensor_jpg, _ = auto_padding(jpeg, patch_size)
    tensor_xyz, _ = auto_padding(linear_space_input, patch_size)
    tensor_jpg, tensor_xyz = tensor_jpg.squeeze(), tensor_xyz.squeeze()
    print("tensor jpg size", tensor_jpg.shape)
    print("tensor xyz size", tensor_xyz.shape)
    rgb_padded = F.pad(
        tensor_jpg, [(neighbourhood_size-patch_size)//2]*4, mode="reflect")
    xyz_padded = F.pad(
        tensor_xyz,  [(neighbourhood_size-patch_size)//2]*4, mode="reflect")
    # print("padded tensor jpg size", rgb_padded.shape)
    # print("padded tensor xyz size", xyz_padded.shape)

    num_of_patch_x = tensor_jpg.shape[1] // patch_size
    num_of_patch_y = tensor_jpg.shape[2] // patch_size

    def to_pixel_index_l(i): return i*patch_size
    def to_pixel_index_r(i): return (i-1)*patch_size + neighbourhood_size

    # rgb_neighbor_patch_list = []
    # xyz_neighbor_patch_list = []
    with torch.no_grad():
        rec_xyz = torch.zeros_like(tensor_xyz)
        weight_list = []
        for i, j in tqdm.tqdm(list(product(range(num_of_patch_x), range(num_of_patch_y)))):
            # print("i:", i)
            # print("j:", j)
            # print(to_pixel_index_l(j))
            # print(to_pixel_index_r(j+1))
            rgb_neighbor_patch = rgb_padded[:, to_pixel_index_l(i):to_pixel_index_r(
                i+1), to_pixel_index_l(j):to_pixel_index_r(j+1)]

            xyz_neighbor_patch = xyz_padded[:, to_pixel_index_l(i):to_pixel_index_r(
                i+1), to_pixel_index_l(j):to_pixel_index_r(j+1)]
            # rgb_neighbor_patch_list.append(rgb_neighbor_patch)
            # xyz_neighbor_patch_list.append(xyz_neighbor_patch)

            rec_xyz_patch, weight = solve_linear(
                rgb_neighbor_patch, xyz_neighbor_patch, sampling_ratio, neighbourhood_size, patch_size)
            weight_list.append(weight)
            rec_xyz[:, i*patch_size:(i+1)*patch_size, j *
                    patch_size:(j+1)*patch_size] = rec_xyz_patch

        # sampling_ratio = round(math.sqrt(1/desired_ratio))
        # bpp = 1/(sampling_ratio**2)*3*16
        # print("BPP: %.4e" % bpp)

        rec_xyz = rec_xyz.unsqueeze(0)
        # output = rec_xyz.permute(0, 2, 3, 1).cpu().data.numpy()
        # output = np.minimum(np.maximum(output, 0), 1)
        # output = output[0, :, :, :]
        # Image.fromarray((output*255).astype('uint8')
        #                 ).save(result_dir + 'rec.png')

        rec_xyz = rec_xyz.clip(0, 1)
        save_image(rec_xyz, 'rec.png')
        return rec_xyz


def SID_model(rec_xyz_tensor, test_id, input_id, mode):
    model = SeeInDark()
    model.load_state_dict(torch.load(
        m_path + m_name, map_location=torch.device('cpu')))
    model = model.to(device)
    # os.makedirs(result_dir)
    if mode == 'lossless':
        print("current SID mode: lossless")
        result_path = glob.glob(result_dir)
    elif mode == 'compress':
        print("current SID mode: compress")
        result_path = glob.glob(compress_result_dir)
    else:
        print('Invalid mode')

    # test the first image in each sequence
    in_path = input_id
    _, in_fn = os.path.split(in_path)
    # print(in_fn)
    gt_path = gt_dir + '%05d_00_10s.ARW' % test_id
    _, gt_fn = os.path.split(gt_path)
    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure/in_exposure, 300)

    ###
    raw = rawpy.imread(in_path)
    ###

    im = raw.raw_image_visible.astype(np.float32)
    input_full = np.expand_dims(pack_raw(im), axis=0) * ratio

    # check main_batch.py for this function
    # im = raw.postprocess(use_camera_wb=True, half_size=False,
    # no_auto_bright=True)
    im = raw.postprocess(use_camera_wb=True, half_size=False,
                         no_auto_bright=True, output_bps=16)
    scale_full = np.expand_dims(np.float32(im/65535.0), axis=0)

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(
        use_camera_wb=True, half_size=False, no_auto_bright=True)
    gt_full = np.expand_dims(np.float32(im/65535.0), axis=0)

    input_full = np.minimum(input_full, 1.0)

    in_img = torch.from_numpy(
        input_full).permute(0, 3, 1, 2).to(device)

    ###
    with torch.no_grad():
        # rec_xyz_tensor = rec_xyz_tensor.permute(0, 1, 2).to(device)
        # rec_xyz_tensor = torch.unsqueeze(rec_xyz_tensor, dim=0)
        print(rec_xyz_tensor.shape)
        out_img = model(rec_xyz_tensor)
    ###

    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]
    # gt_full = gt_full[0, :, :, :]
    scale_full = scale_full[0, :, :, :]
    origin_full = scale_full
    # scale the low-light image to the same mean of the groundtruth
    scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full)

    Image.fromarray((origin_full*255).astype('uint8')
                    ).save(result_path[0] + '/%5d_00_%d_ori.png' % (test_id, ratio))
    Image.fromarray((output*65535).astype('uint8')
                    ).save(result_path[0] + '/%5d_00_%d_out.png' % (test_id, ratio))
    Image.fromarray((scale_full*255).astype('uint8')
                    ).save(result_path[0] + '/%5d_00_%d_scale.png' % (test_id, ratio))
    # Image.fromarray((gt_full*255).astype('uint8')
    #                 ).save(result_path[0] + '/%5d_00_%d_gt.png' % (test_id, ratio))


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def solve_linear_noM(rgb_neighbor_patch, xyz_neighbor_patch, sampling_ratio, neighbourhood_size, patch_size):
    x = np.arange(0, rgb_neighbor_patch.shape[1], sampling_ratio)
    y = np.arange(0, rgb_neighbor_patch.shape[2], sampling_ratio)
    Y, X = np.meshgrid(y, x)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X_rescale = X / rgb_neighbor_patch.shape[1]
    Y_rescale = Y / rgb_neighbor_patch.shape[2]
    # to_pil_image(xyz_neighbor_patch[:, X, Y])
    Y_full, X_full = np.meshgrid(
        np.arange(0, rgb_neighbor_patch.shape[2], 1), np.arange(0, rgb_neighbor_patch.shape[1], 1))
    S_full = torch.cat([rgb_neighbor_patch, torch.tensor(
        X_full/rgb_neighbor_patch.shape[1]).unsqueeze(0), torch.tensor(Y_full/rgb_neighbor_patch.shape[2]).unsqueeze(0)], dim=0)

    S = torch.cat([rgb_neighbor_patch[:, X, Y], X_rescale.unsqueeze(0),
                   Y_rescale.unsqueeze(0)], dim=0)

    S_flattened = S.view(S.shape[0], -1)
    M = torch.norm(S_flattened.unsqueeze(
        2) - S_flattened.unsqueeze(1), dim=0)

    r_c = xyz_neighbor_patch[:, X, Y]
    P = torch.cat([torch.ones([M.shape[0], 1]), S_flattened.T], dim=1)

    A = P
    B = r_c.view(r_c.shape[0], -1).permute(1, 0)

    res = torch.linalg.solve(A, B)

    # For inference
    index_min, index_max = (
        neighbourhood_size - patch_size)//2, (neighbourhood_size + patch_size)//2
    query_patch = S_full[:, index_min:index_max,
                         index_min:index_max].contiguous()
    S_query_flattened = query_patch.view(S_full.shape[0], -1)
    M_ = torch.norm(S_query_flattened.unsqueeze(
        2) - S_flattened.unsqueeze(1), dim=0)
    P_ = torch.cat([torch.ones([M_.shape[0], 1]),
                    S_query_flattened.T], dim=1)
    rec_xyx_patch = (P_@res.double()).reshape(100, 100, 4).permute(2, 0, 1)
    return rec_xyx_patch


def solve_linear(rgb_neighbor_patch, xyz_neighbor_patch, sampling_ratio, neighbourhood_size, patch_size):
    x = np.arange(0, rgb_neighbor_patch.shape[1], sampling_ratio)
    y = np.arange(0, rgb_neighbor_patch.shape[2], sampling_ratio)
    Y, X = np.meshgrid(y, x)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X_rescale = X / rgb_neighbor_patch.shape[1]
    Y_rescale = Y / rgb_neighbor_patch.shape[2]
    # to_pil_image(xyz_neighbor_patch[:, X, Y])
    Y_full, X_full = np.meshgrid(
        np.arange(0, rgb_neighbor_patch.shape[2], 1), np.arange(0, rgb_neighbor_patch.shape[1], 1))
    S_full = torch.cat([rgb_neighbor_patch, torch.tensor(
        X_full/rgb_neighbor_patch.shape[1]).unsqueeze(0), torch.tensor(Y_full/rgb_neighbor_patch.shape[2]).unsqueeze(0)], dim=0)

    S = torch.cat([rgb_neighbor_patch[:, X, Y], X_rescale.unsqueeze(0),
                   Y_rescale.unsqueeze(0)], dim=0)

    S_flattened = S.view(S.shape[0], -1)
    M = torch.norm(S_flattened.unsqueeze(
        2) - S_flattened.unsqueeze(1), dim=0)

    gamma = torch.randn([3, M.shape[0]])
    beta = torch.randn([3, 6])
    # print("xyz_neighbor_patch shape:", xyz_neighbor_patch.shape)
    r_c = xyz_neighbor_patch[:, X, Y]
    # print("r_c shape: ", r_c.shape)
    P = torch.cat([torch.ones([M.shape[0], 1]), S_flattened.T], dim=1)

    A = torch.cat([torch.cat([M, P], dim=1), torch.cat(
        [P.T, torch.zeros(6, 6)], dim=1)], dim=0)
    B = torch.cat([r_c.view(r_c.shape[0], -1).permute(1, 0),
                   torch.zeros(6, r_c.shape[0])], dim=0)
    A = A.float()
    B = B.float()
    try:
        res = torch.linalg.solve(A, B)
    except:
        res = torch.linalg.solve(A+torch.rand_like(A)*1e-3, B)
    # res = torch.linalg.solve(A, B)

    # For inference
    index_min, index_max = (
        neighbourhood_size - patch_size)//2, (neighbourhood_size + patch_size)//2
    query_patch = S_full[:, index_min:index_max,
                         index_min:index_max].contiguous()
    S_query_flattened = query_patch.view(S_full.shape[0], -1)
    M_ = torch.norm(S_query_flattened.unsqueeze(
        2) - S_flattened.unsqueeze(1), dim=0)
    P_ = torch.cat([torch.ones([M_.shape[0], 1]),
                    S_query_flattened.T], dim=1)
    rec_xyx_patch = (
        torch.cat([M_, P_], dim=1)@res.double()).reshape(128, 128, 3).permute(2, 0, 1)
    return rec_xyx_patch, res


if __name__ == "__main__":
    # tensor_jpg, tensor_xyz = main_batch_preprocess()
    # print("output tensor size: ", tensor_xyz.shape)
    counter = 0
    test_ids.sort()
    mode = 'lossless'
    # print(test_ids)
    for test_id in test_ids:
        for i in range(3):
            img_index = counter + i
            tensor_jpg, tensor_xyz, input_id = reconstruct_preprocess(
                img_index, mode, test_id)
            rec_xyz_tensor = reconstrcut(
                tensor_jpg, tensor_xyz, desired_ratio=0.0002)
            print('tensor_xyz shape: ', tensor_xyz.shape)
            print('rec_xyz_tensor shape: ', rec_xyz_tensor.shape)
            permute_tensor = RGB_to_Bayer(rec_xyz_tensor)
            print('permuted shape: ', permute_tensor.shape)
            # test_id is GT_id for SID
            SID_model(permute_tensor, test_id, input_id, mode)
        counter += 3


# pass
