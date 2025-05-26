import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_dice_coefficient(mask1, mask2):
    """计算两个mask之间的Dice系数"""
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice = (2. * intersection) / (union + 1e-6)  # 添加小值避免除以零

    return dice

def analyze_noise_effects(ground_truth_dir, predict_dir):
    """分析不同噪声水平对Dice系数的影响"""
    results = []

    # 获取所有预测文件
    pred_files = [f for f in os.listdir(predict_dir) if f.endswith('.nii.gz')]
    print(f"找到 {len(pred_files)} 个预测文件")

    for pred_file in pred_files:
        # 解析文件名获取患者ID和噪声水平
        filename = os.path.splitext(os.path.splitext(pred_file)[0])[0]  # 移除.nii.gz
        parts = filename.split('_')
        patient_id = parts[0] + '_' + parts[1]
        noise_level = float(parts[-1])

        # 构建对应的ground truth文件名
        gt_file = f"{patient_id}.nii.gz"
        gt_path = os.path.join(ground_truth_dir, gt_file)

        if not os.path.exists(gt_path):
            print(f"警告: 找不到GT文件 {gt_file}，跳过")
            continue

        # 加载图像数据
        gt_img = nib.load(gt_path).get_fdata()
        pred_img = nib.load(os.path.join(predict_dir, pred_file)).get_fdata()

        # 计算Dice系数
        dice = calculate_dice_coefficient(gt_img, pred_img)

        # 记录结果
        results.append({
            'Patient ID': patient_id,
            'Noise Level': noise_level,
            'Dice': dice
        })

    return pd.DataFrame(results)


def visualize_results(df):
    """可视化分析结果"""
    plt.figure(figsize=(10, 6))

    # Create the line plot with error bands
    sns.lineplot(data=df, x='Noise Level', y='Dice', marker='o',
                 ci='sd', err_style='band', color='b')

    # Calculate mean values for each noise level
    mean_values = df.groupby('Noise Level')['Dice'].mean().reset_index()

    # Add text annotations for each point
    for index, row in mean_values.iterrows():
        plt.text(x=row['Noise Level'],
                 y=row['Dice'] + 0.02,  # 适当调整，防止遮挡
                 s=f"{row['Dice']:.2f}",
                 ha='center',
                 va='bottom',
                 fontsize=10,
                 color='black')
    plt.title('Average Dice Coefficient vs Noise Level\n(Mean ± STD)')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Dice Coefficient')
    plt.ylim(0.4, 1.05)

    # Adjust layout and save
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('mean_dice_coefficient_pepper.png', dpi=300)
    plt.show()


def main():
    # 配置路径
    ground_truth_dir = "E:/Ultrasound_data/Dicom_file/MedicalNet/labels/"
    predict_dir = "E:\\Ultrasound_data\\Dicom_file\\Pepper_noise\\seg_pepper\\dcm_pepper\\"

    # 检查路径是否存在
    if not os.path.exists(ground_truth_dir):
        print(f"错误: GT路径不存在 - {ground_truth_dir}")
        return
    if not os.path.exists(predict_dir):
        print(f"错误: Predict路径不存在 - {predict_dir}")
        return

    # 分析数据
    print("开始分析噪声效应...")
    df = analyze_noise_effects(ground_truth_dir, predict_dir)

    if df.empty:
        print("错误: 没有分析到任何有效数据，请检查输入路径和文件")
        return

    # 计算统计信息
    print("\n=== 统计结果 ===")
    stats = df.groupby('Noise Level')['Dice'].agg(['mean', 'std', 'count'])
    print(stats)

    # 保存结果
    #df.to_csv('noise_dice_results.csv', index=False)
    #print("\n结果已保存到 noise_dice_results.csv")

    # 可视化
    visualize_results(df)


if __name__ == "__main__":
    main()
