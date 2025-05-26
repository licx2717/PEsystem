from torchvision.transforms import functional as TF
import os
import nibabel as nib
import numpy as np
import torch

def add_rotation(image, angle):
    """
    对图像进行旋转
    Args:
        image (torch.Tensor): 输入图像（C, H, W）
        angle (float): 旋转角度
    Returns:
        torch.Tensor: 旋转后的图像
    """
    return TF.rotate(image, angle)

def process_nii_gz_with_rotation(input_path, output_dir, angles):
    """
    读取 .nii.gz 文件，增加不同角度的旋转并保存
    Args:
        input_path (str): 输入 .nii.gz 文件路径
        output_dir (str): 输出文件夹路径
        angles (list): 旋转角度列表，例如 [0, 10, 20, 30, ...]
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载图像
    nii_img = nib.load(input_path)
    image_data = nii_img.get_fdata()

    # 将图像数据归一化到 [0, 1] 范围
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # 对所有切片进行旋转
    for angle in angles:
        # 创建一个空数组来存储旋转后的体积数据
        rotated_volume = np.zeros_like(image_data)

        # 遍历每个切片并应用旋转
        for slice_idx in range(image_data.shape[-1]):
            # 获取当前切片
            slice_data = image_data[:, :, slice_idx]

            # 将 NumPy 数组转换为 PyTorch Tensor
            slice_tensor = torch.from_numpy(slice_data).float()
            slice_tensor = slice_tensor.unsqueeze(0)  # 添加 channel 维度 [1, H, W]

            # 应用旋转
            rotated_slice = add_rotation(slice_tensor, angle)

            # 将旋转后的切片存储到旋转后的体积数据中
            rotated_volume[:, :, slice_idx] = rotated_slice.squeeze().numpy()

        # 保存旋转后的图像
        output_path = os.path.join(output_dir, f'rotated_angle_{angle}.nii.gz')
        rotated_nii = nib.Nifti1Image(rotated_volume, nii_img.affine, nii_img.header)
        nib.save(rotated_nii, output_path)
        print(f"Saved rotated image at angle {angle} to {output_path}")

# 参数设置
input_path = "D:\\MONAI-dev\\testing\\robustness\\picture\\N_00020.nii.gz"  # 输入 .nii.gz 文件路径
output_dir = "D:\\MONAI-dev\\testing\\robustness\\"  # 输出文件夹路径
angles = [0, 10, 20, 30, 40, 50, 60]  # 旋转角度列表

# 运行处理
process_nii_gz_with_rotation(input_path, output_dir, angles)
