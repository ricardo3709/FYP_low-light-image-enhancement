from locale import normalize
from unittest.mock import patch
import numpy as np
from dataset import RawImageDataset
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
from PIL import Image

from io import BytesIO


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

# print("PSNR: %.4f" % compute_psnr(tensor_jpg[0], to_tensor(rgb_jpeg)))


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
    rec_xyx_patch = (P_@res.double()).reshape(100, 100, 3).permute(2, 0, 1)
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
    r_c = xyz_neighbor_patch[:, X, Y]
    P = torch.cat([torch.ones([M.shape[0], 1]), S_flattened.T], dim=1)

    A = torch.cat([torch.cat([M, P], dim=1), torch.cat(
        [P.T, torch.zeros(6, 6)], dim=1)], dim=0)
    B = torch.cat([r_c.view(r_c.shape[0], -1).permute(1, 0),
                   torch.zeros(6, r_c.shape[0])], dim=0)

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
        torch.cat([M_, P_], dim=1)@res.double()).reshape(100, 100, 3).permute(2, 0, 1)
    return rec_xyx_patch, res


def reconstrcut(jpeg, linear_space_input, patch_size=100, neighbourhood_size=500, desired_ratio=0.002):
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
    tensor_jpg, _ = auto_padding(jpeg, patch_size)
    tensor_xyz, _ = auto_padding(linear_space_input, patch_size)
    tensor_jpg, tensor_xyz = tensor_jpg.squeeze(), tensor_xyz.squeeze()
    rgb_padded = F.pad(
        tensor_jpg, [(neighbourhood_size-patch_size)//2]*4, mode="reflect")
    xyz_padded = F.pad(
        tensor_xyz,  [(neighbourhood_size-patch_size)//2]*4, mode="reflect")

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
        rec_xyz = rec_xyz.clip(0, 1)
        sampling_ratio = round(math.sqrt(1/desired_ratio))
        bpp = 1/(sampling_ratio**2)*3*16
        print("BPP: %.4e" % bpp)
        # save_image([rec_xyz, tensor_xyz, (rec_xyz-tensor_xyz).abs()], 't.jpg')
        # save_image(torch.stack(weight_list, dim=0).view(num_of_patch_x, num_of_patch_y, *weight_list[0].shape).permute(2,3,0,1), "weight.jpg", normalize=True, scale_each=True)
        save_image(rec_xyz, "rec.png")
        return rec_xyz, tensor_xyz


if __name__ == "__main__":
    dataset = RawImageDataset("./datasets/SID", split="train")
    #dataset = RawImageDataset("./datasets/fivek_dataset", split="val")
    # for i in range(5):
    psnr_list = []
    for i in range(5):
        _, raw_file, raw_path = dataset[i]

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

        rec_xyz, tensor_xyz = reconstrcut(
            tensor_jpg, tensor_xyz, desired_ratio=0.0002)
        psnr = compute_psnr(rec_xyz, tensor_xyz)
        print("PSNR: %.4f" % psnr)
        psnr_list.append(psnr)
    print(np.mean(psnr_list))

    # pass
