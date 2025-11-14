#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_pytorch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """用 PyTorch 计算 SSIM（高效版）"""
    channel = img1.shape[1]
    padding = window_size // 2
    gaussian_window = torch.ones((channel, 1, window_size, window_size), device=img1.device) / (window_size**2)

    mu1 = F.conv2d(img1, gaussian_window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, gaussian_window, padding=padding, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, gaussian_window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, gaussian_window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, gaussian_window, padding=padding, groups=channel) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=1e-6)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-6)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map



import torch.nn.functional as F

def compute_gradient_loss(pred, gt):
    """计算梯度损失，减少模糊"""
    def gradient(x):
        grad_x = x[:, :-1, :] - x[:, 1:, :]
        grad_y = x[:, :, :-1] - x[:, :, 1:]
        return grad_x, grad_y
    
    pred_grad_x, pred_grad_y = gradient(pred)
    gt_grad_x, gt_grad_y = gradient(gt)
    
    loss_x = F.l1_loss(pred_grad_x, gt_grad_x)
    loss_y = F.l1_loss(pred_grad_y, gt_grad_y)
    
    return loss_x + loss_y

def tile_based_ssim_loss(image, gt_image, tile_size=16, gamma=-2, lambda_dssim=0.8, beta=0.1):
    """优化版：使用 unfold() 提取所有 tile，提升效率（单张图片）"""
    C, H, W = image.shape

    # 用 unfold 提取 tile
    unfolded_img = F.unfold(image.unsqueeze(0), kernel_size=tile_size, stride=tile_size)  # (B, C, tile_size*tile_size, num_tiles)
    unfolded_gt = F.unfold(gt_image.unsqueeze(0), kernel_size=tile_size, stride=tile_size)

    unfolded_img = unfolded_img.view(1, C, tile_size, tile_size, -1).permute(0, 4, 1, 2, 3)  # 将 tile 转为合适的维度
    unfolded_gt = unfolded_gt.view(1, C, tile_size, tile_size, -1).permute(0, 4, 1, 2, 3)

    # 计算 SSIM
    ssim_map = ssim_pytorch(unfolded_img.squeeze(0), unfolded_gt.squeeze(0))
    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # 对所有 tile 取最小的那部分 SSIM

    # 计算 tile 权重
    tile_weight = torch.exp(gamma * (ssim_val - 1))
    
    # 计算 L1 Loss
    l1_loss = F.l1_loss(unfolded_img, unfolded_gt, reduction="none").mean(dim=(2, 3, 4))

    # 计算梯度损失
    grad_loss = compute_gradient_loss(unfolded_img.squeeze(0), unfolded_gt.squeeze(0))

    # 计算整体 loss
    tile_loss =  tile_weight * (lambda_dssim * (1 - ssim_val) + beta * grad_loss)
    loss = tile_loss.mean()  # 归一化
    
    return loss
