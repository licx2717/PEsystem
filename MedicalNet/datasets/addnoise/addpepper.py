import os
import numpy as np
import SimpleITK as sitk
import random


def add_salt_pepper_noise(image_array, salt_prob=0.05, pepper_prob=0.05):
    """
    向图像数组添加椒盐噪声
    Args:
        image_array: 输入图像数组
        salt_prob: 盐噪声概率 (白色像素)
        pepper_prob: 椒噪声概率 (黑色像素)
    Returns:
        添加椒盐噪声后的图像数组
    """
    noisy_image = np.copy(image_array)

    # 获取图像的最小和最大值
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    # 添加盐噪声 (最大值)
    salt_mask = np.random.random(image_array.shape) < salt_prob
    noisy_image[salt_mask] = max_val

    # 添加椒噪声 (最小值)
    pepper_mask = np.random.random(image_array.shape) < pepper_prob
    noisy_image[pepper_mask] = min_val

    return noisy_image


def normalize(volume):
    """归一化图像数据到0-1范围"""
    min_val = np.min(volume)
    max_val = np.max(volume)
    volume = (volume - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else volume
    return volume


def reduce_dimensions(data):
    """降低数据维度"""
    if np.ndim(data) == 5:
        data = data[:, :, :, :, 0]  # 5D -> 4D
    elif np.ndim(data) == 4:
        data = data[:, :, :, 0]  # 4D -> 3D
    return data


def data_transform_with_salt_pepper_noise(input_folder, output_folder, noise_levels=[0.05]):
    """
    处理nii.gz文件，添加椒盐噪声后裁剪和重采样
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        noise_levels: 椒盐噪声水平列表(总噪声概率 = 盐噪声 + 椒噪声)
    """
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".dcm"):
                # 读取原始图像
                input_path = os.path.join(root, file)
                try:
                    image = sitk.ReadImage(input_path)
                    image_array = sitk.GetArrayFromImage(image)

                    # 对每个噪声水平处理
                    for noise_level in noise_levels:
                        # 将噪声水平分配给盐和椒噪声(各一半)
                        salt_prob = noise_level / 2
                        pepper_prob = noise_level / 2

                        # 添加椒盐噪声
                        noisy_array = add_salt_pepper_noise(
                            image_array,
                            salt_prob=salt_prob,
                            pepper_prob=pepper_prob
                        )

                        # 从噪声数据创建新图像
                        noisy_image = sitk.GetImageFromArray(noisy_array)
                        noisy_image.CopyInformation(image)  # 保留原始图像信息

                        # 1. 裁剪图像
                        start_index = [100, 50, 0]
                        size = [600, 500, 48]
                        roi_filter = sitk.RegionOfInterestImageFilter()
                        roi_filter.SetIndex(start_index)
                        roi_filter.SetSize(size)

                        try:
                            cropped_image = roi_filter.Execute(noisy_image)
                        except RuntimeError:
                            cropped_image = noisy_image
                            print(f"WARNING: {file} - 文件大小不适合裁剪，跳过裁剪步骤")

                        # 2. 重采样
                        new_spacing = [1, 1, 1]
                        new_size = [224, 224, 16]
                        resample = sitk.ResampleImageFilter()
                        resample.SetOutputSpacing(new_spacing)
                        resample.SetSize(new_size)
                        resampled_image = resample.Execute(cropped_image)

                        # 3. 归一化和降维
                        processed_array = sitk.GetArrayFromImage(resampled_image)
                        normalized_array = normalize(processed_array)
                        final_array = reduce_dimensions(normalized_array)
                        final_image = sitk.GetImageFromArray(final_array)

                        # 保存结果
                        base_name = os.path.splitext(os.path.splitext(file)[0])[0]  # 去除.nii.gz
                        output_name = f"{base_name}_saltpepper_{noise_level:.2f}.nii.gz"
                        output_path = os.path.join(output_folder, output_name)
                        sitk.WriteImage(final_image, output_path)

                        print(f"处理完成: {file} -> {output_name} (椒盐噪声水平 {noise_level:.2f})")
                        print(f"最终图像大小: {final_image.GetSize()}")

                except Exception as e:
                    print(f"处理文件 {file} 时出错: {str(e)}")
                    continue


# 使用示例
input_folder = "E:\\Ultrasound_data\\Dicom_file\\MedicalNet\\original\\"
output_folder = "E:\\Ultrasound_data\\Dicom_file\\Pepper_noise\\dcm_pepper\\"
noise_levels = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]  # 设置椒盐噪声级别

data_transform_with_salt_pepper_noise(input_folder, output_folder, noise_levels)
