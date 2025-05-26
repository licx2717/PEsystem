import os
import numpy as np
import torch
import nibabel as nib

def add_salt_pepper_noise(image, prob=0.05):
    """
    添加椒盐噪声
    :param image: 输入图像，假设为归一化后的张量，范围在 [0, 1]
    :param salt_prob: 盐噪声的概率
    :param pepper_prob: 椒噪声的概率
    :return: 添加椒盐噪声后的图像
    """
    # 确保输入图像是 PyTorch 张量
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # 创建一个与输入图像形状相同的噪声矩阵
    noise = torch.zeros_like(image)

    # 添加盐噪声（白色像素，值为 1）
    salt_mask = torch.rand_like(image) < prob
    noise[salt_mask] = 1.0

    # 添加椒噪声（黑色像素，值为 0）
    pepper_mask = torch.rand_like(image) < prob
    noise[pepper_mask] = 0.0

    # 将噪声添加到原始图像
    noisy_image = image + noise

    # 确保图像的像素值在 [0, 1] 范围内
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image

def process_nii_gz(input_path, output_dir, noise_levels):
    """
    读取 .nii.gz 文件，添加椒盐噪声并保存
    Args:
        input_path (str): 输入 .nii.gz 文件路径
        output_dir (str): 输出文件夹路径
        noise_levels (dict): 椒盐噪声的概率，例如 {'salt_pepper': [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]}
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载图像
    nii_img = nib.load(input_path)
    image_data = nii_img.get_fdata()

    # 将图像数据归一化到 [0, 1] 范围
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # 将 NumPy 数组转换为 PyTorch 张量，并扩展为 4D 张量 (batch_size=1, channels=1, height, width)
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).float()

    # 添加不同强度的椒盐噪声并保存
    for prob in noise_levels['salt_pepper']:
        noisy_image_tensor = add_salt_pepper_noise(image_tensor, prob)

        # 将噪声图像恢复原始范围（假设原始图像的范围在归一化前已知）
        noisy_image = noisy_image_tensor.squeeze().numpy()
        noisy_image = noisy_image * (np.max(image_data) - np.min(image_data)) + np.min(image_data)

        # 创建 NIfTI 图像对象
        noisy_nii_img = nib.Nifti1Image(noisy_image, affine=nii_img.affine)

        # 保存为 .nii.gz 文件
        output_path = os.path.join(output_dir, f'salt_pepper_image_prob{prob:.2f}.nii.gz')
        nib.save(noisy_nii_img, output_path)
        print(f"Saved noisy image with prob={prob:.2f} to {output_path}")

# 参数设置
input_path = "D:\\MONAI-dev\\testing\\robustness\\picture\\N_00020.nii.gz"  # 输入 .nii.gz 文件路径
output_dir = "D:\\MONAI-dev\\testing\\robustness\\"  # 输出文件夹路径
noise_levels = {'salt_pepper': [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]}  # 椒盐噪声的概率

# 运行处理
process_nii_gz(input_path, output_dir, noise_levels)
