import os
import numpy as np
import torch
import nibabel as nib


def add_gaussian_noise(image, std):
    """
    给图像添加高斯噪声
    Args:
        image (np.ndarray): 输入图像
        std (float): 高斯噪声的标准差
    Returns:
        np.ndarray: 添加噪声后的图像
    """
    noise = np.random.normal(loc=0, scale=std, size=image.shape)
    noisy_image = image + noise
    # 将值裁剪到 [0, 1] 范围（假设图像值在 [0, 1] 范围内）
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def process_nii_gz(input_path, output_dir, noise_levels):
    """
    读取 .nii.gz 文件，添加高斯噪声并保存
    Args:
        input_path (str): 输入 .nii.gz 文件路径
        output_dir (str): 输出文件夹路径
        noise_levels (dict): 高斯噪声的标准差，例如 {'gaussian': [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]}
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载图像
    nii_img = nib.load(input_path)
    image_data = nii_img.get_fdata()

    # 将图像数据归一化到 [0, 1] 范围
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # 添加不同强度的高斯噪声并保存
    for std in noise_levels['gaussian']:
        noisy_image = add_gaussian_noise(image_data, std)

        # 将噪声图像恢复原始范围（假设原始图像的范围在归一化前已知）
        noisy_image = noisy_image * (np.max(image_data) - np.min(image_data)) + np.min(image_data)

        # 创建 NIfTI 图像对象
        noisy_nii_img = nib.Nifti1Image(noisy_image, affine=nii_img.affine)

        # 保存为 .nii.gz 文件
        output_path = os.path.join(output_dir, f'noisy_image_std{std:.2f}.nii.gz')
        nib.save(noisy_nii_img, output_path)
        print(f"Saved noisy image with std={std:.2f} to {output_path}")


# 参数设置
input_path = "D:\\MONAI-dev\\testing\\robustness\\N_00020.nii.gz"  # 输入 .nii.gz 文件路径
output_dir = "D:\\MONAI-dev\\testing\\robustness\\"  # 输出文件夹路径
noise_levels = {'gaussian': [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]}  # 高斯噪声的标准差

# 运行处理
process_nii_gz(input_path, output_dir, noise_levels)
