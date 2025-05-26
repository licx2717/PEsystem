import os
import torch
from monai.transforms import Rand3DElastic
import nibabel as nib
import numpy as np

def add_elastic_deformation(image, alpha, sigma, save_dir=None, file_name=None):
    """
    使用 MONAI 的 Rand3DElastic 实现弹性形变
    Args:
        image (torch.Tensor): 输入图像 (C, H, W) 或 (C, D, H, W)
        alpha (float): 形变强度
        sigma (float): 高斯滤波的标准差
        save_dir (str): 保存图像的目录，如果为 None 则不保存
        file_name (str): 保存文件的名称，如果为 None 则不保存
    Returns:
        torch.Tensor: 形变后的图像
    """
    # Step 1: 确保输入图像维度正确 [batch, C, D, H, W]
    while image.dim() < 5:
        image = image.unsqueeze(0)  # 补全到 5D [batch, C, D, H, W]

    # Step 2: 提取空间维度 (D, H, W)
    depth, height, width = image.shape[-3:]

    # Step 3: 设置 spatial_size 为 (D, H, W)
    spatial_size = (depth, height, width)

    # Step 4: 定义 MONAI 的 Rand3DElastic 转换
    transform = Rand3DElastic(
        sigma_range=(sigma, sigma),
        magnitude_range=(alpha, alpha),
        prob=1.0,
        spatial_size=spatial_size,  # 必须为 (D, H, W)
        mode="bilinear",
    )

    # Step 5: 应用形变
    image = image.squeeze(0)
    deformed_image = transform(image)

    # Step 6: 保存形变后的图像
    if save_dir is not None and file_name is not None:
        os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
        save_path = os.path.join(save_dir, file_name)

        # 将图像数据转换为 NumPy 数组
        deformed_image_np = deformed_image.cpu().numpy()[0]  # 提取 [C, D, H, W] 的 [0] 通道
        deformed_image_np = deformed_image_np.astype(np.float32)  # 确保数据类型为 float32

        # 创建 NIfTI 图像对象
        affine = np.eye(4)  # 默认仿射矩阵
        nii_img = nib.Nifti1Image(deformed_image_np, affine)

        # 保存为 .nii.gz 文件
        nib.save(nii_img, save_path)
        print(f"[INFO] Saved deformed image to {save_path}")

    # Step 7: 压缩多余的 batch 维度 (如果有)
    return deformed_image

def process_nii_gz_with_elastic_deformation(input_path, output_dir, alpha_sigma_pairs):
    """
    读取 .nii.gz 文件，添加弹性形变并保存
    Args:
        input_path (str): 输入 .nii.gz 文件路径
        output_dir (str): 输出文件夹路径
        alpha_sigma_pairs (list): 形变参数列表，例如 [(alpha1, sigma1), (alpha2, sigma2), ...]
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载图像
    nii_img = nib.load(input_path)
    image_data = nii_img.get_fdata()

    # 将图像数据归一化到 [0, 1] 范围
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # 将 NumPy 数组转换为 PyTorch Tensor
    image_tensor = torch.from_numpy(image_data).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度 [1, 1, D, H, W]

    # 添加不同强度的弹性形变并保存
    for alpha, sigma in alpha_sigma_pairs:
        # 应用弹性形变
        deformed_image = add_elastic_deformation(image_tensor, alpha, sigma, save_dir=output_dir, file_name=f'deformed_alpha_{alpha}_sigma_{sigma}.nii.gz')

# 参数设置
input_path = "D:\\MONAI-dev\\testing\\robustness\\picture\\N_00020.nii.gz"  # 输入 .nii.gz 文件路径
output_dir = "D:\\MONAI-dev\\testing\\robustness\\"  # 输出文件夹路径
alpha_sigma_pairs = [(0, 3), (10, 3), (20, 3), (30, 3), (40, 3), (50, 3),(60,3)]  # 形变参数列表

# 运行处理
process_nii_gz_with_elastic_deformation(input_path, output_dir, alpha_sigma_pairs)
