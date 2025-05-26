import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_nii_file(filepath):
    img = nib.load(filepath)
    img = img.get_fdata()
    return img


def compute_slicewise_dice(mask1, mask2):
    """计算每层slice的Dice系数"""
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    slice_dice = []

    for z in range(mask1.shape[2]):  # 假设z轴是slice方向
        slice1 = mask1[:, :, z]
        slice2 = mask2[:, :, z]

        # 确保slice中有至少一个mask有非零值
        if np.any(slice1) or np.any(slice2):
            intersection = np.sum(slice1 * slice2)
            volumes = np.sum(slice1) + np.sum(slice2)
            if volumes == 0:
                dice = 0
            else:
                dice = (2. * intersection) / volumes
            slice_dice.append(dice)

    return slice_dice


folder_A = r"E:\\Ultrasound_data\\Dicom_file\\Slice_stability\\labels\\"

folder_B = r"E:\\Ultrasound_data\\Dicom_file\\Slice_stability\\segmentation\\"

all_slice_dice = []  # 存储所有slice的Dice系数
file_slice_stats = []  # 存储每个文件的slice Dice统计信息

for root, dirs, files in os.walk(folder_B):
    for file in files:
        matching_file_in_A = os.path.join(folder_A, file)
        if os.path.exists(matching_file_in_A) and matching_file_in_A.endswith('.nii.gz'):
            mask1 = load_nii_file(matching_file_in_A)
            mask2 = load_nii_file(os.path.join(root, file))

            # 计算每层slice的Dice系数
            slice_dice = compute_slicewise_dice(mask1, mask2)
            all_slice_dice.extend(slice_dice)

            # 计算当前文件的slice Dice统计信息
            if slice_dice:
                file_mean = np.mean(slice_dice)
                file_std = np.std(slice_dice)
                file_min = np.min(slice_dice)
                file_max = np.max(slice_dice)

                file_slice_stats.append({
                    'Filename': file,
                    'Mean Dice': file_mean,
                    'Std Dice': file_std,
                    'Min Dice': file_min,
                    'Max Dice': file_max,
                    'Num Slices': len(slice_dice)
                })

                print(f"File: {file}")
                print(
                    f"  Slice Dice - Mean: {file_mean:.3f}, Std: {file_std:.3f}, Min: {file_min:.3f}, Max: {file_max:.3f}")

# 分析所有slice的Dice系数
if all_slice_dice:
    overall_mean = np.mean(all_slice_dice)
    overall_std = np.std(all_slice_dice)

    print("\nOverall Slice-wise Dice Statistics:")
    print(f"Mean: {overall_mean:.3f}")
    print(f"Standard Deviation: {overall_std:.3f}")
    print(f"Number of slices analyzed: {len(all_slice_dice)}")

    # 保存统计信息到Excel
    df_stats = pd.DataFrame(file_slice_stats)
    df_stats.to_excel("D:\\Dicom_file\\MedicalNet\\slice_dice_statistics.xlsx", index=False)

    # 设置全局绘图样式
    plt.style.use('seaborn')  # 简洁的学术风格
    plt.rcParams.update({
        'font.size': 10,
        'figure.titlesize': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 600,
        'figure.figsize': (8, 6),  # 标准学术图片尺寸
        'savefig.format': 'tiff',  # 学术期刊推荐格式
        'savefig.dpi': 600,
        'font.family': 'serif',
        'font.serif': ['Times New Roman']
    })

    # 1. Dice系数分布直方图 - 用于展示整体准确性和离散程度
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(all_slice_dice, bins=30, edgecolor='white', linewidth=0.5, alpha=0.8,
                                color='#1f77b4', density=False)

    # 添加均值线
    plt.axvline(overall_mean, color='#d62728', linestyle='--', linewidth=1.5,
                label=f'Mean: {overall_mean:.3f}')
    plt.axvline(overall_mean - overall_std, color='#2ca02c', linestyle=':', linewidth=1)
    plt.axvline(overall_mean + overall_std, color='#2ca02c', linestyle=':', linewidth=1,
                label=f'±1 SD')

    plt.xlabel('Dice Similarity Coefficient', fontweight='bold')
    plt.ylabel('Number of Slices', fontweight='bold')
    plt.title('Distribution of Slice-wise Segmentation Accuracy',
              fontweight='bold', pad=10)
    plt.xlim(0.6, 1)
    plt.legend(frameon=False, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    #plt.savefig("slice_dice_nothing.tiff", bbox_inches='tight', dpi=300)
    plt.show()

    # 2. Slice位置-Dice系数曲线图 - 使用Sample1, Sample2等简化图例
    plt.figure(figsize=(8, 6))  # 适合论文全栏宽度

    # 定义不同样本的标记样式和颜色
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h']  # 不同类型的标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # 不同颜色

    # 示例: 随机选择5个有代表性的样本
    sample_files = [s for s in file_slice_stats if s['Num Slices'] >= 5]  # 过滤掉slice太少的文件
    sample_files = np.random.choice(sample_files, min(5, len(sample_files)), replace=False)

    for i, file_stat in enumerate(sample_files):
        file = file_stat['Filename']
        mask1 = load_nii_file(os.path.join(folder_A, file))
        mask2 = load_nii_file(os.path.join(folder_B, file))
        slice_dice = compute_slicewise_dice(mask1, mask2)

        # 使用Sample 1, Sample 2等简化标签
        plt.plot(range(len(slice_dice)), slice_dice,
                 marker=markers[i % len(markers)],  # 循环使用不同标记
                 markersize=5,  # 标记大小
                 markeredgecolor='white',  # 标记边缘白色
                 markeredgewidth=0.5,  # 边缘宽度
                 linestyle='-',
                 linewidth=1.5,
                 color=colors[i % len(colors)],  # 循环使用不同颜色
                 alpha=0.8,
                 label=f"Sample {i + 1} (Mean: {np.mean(slice_dice):.2f})")  # 修改为Sample1, Sample2等

    plt.xlabel('Slice Position (Z-axis)', fontweight='bold')
    plt.ylabel('Dice Similarity Coefficient', fontweight='bold')
    plt.title('Spatial Distribution of Segmentation Accuracy\n(with Individual Slice Markers)',
              fontweight='bold', pad=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.2, linestyle='--')

    # 调整图例位置和样式
    plt.legend(frameon=True,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               framealpha=0.9,
               edgecolor='none',
               title="Samples")  # 添加图例标题

    plt.tight_layout()
    #plt.savefig("spatial_stability_with_nothing.tiff", bbox_inches='tight', dpi=600)
    plt.show()


