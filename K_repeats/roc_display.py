import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def read_excel_files(directory):
    """遍历文件夹，返回所有 Excel 文件的路径"""
    file_paths = [os.path.join(directory, filename)
                  for filename in os.listdir(directory)
                  if filename.endswith('.xlsx') or filename.endswith('.xls')]
    return file_paths

def read_data(file_path):
    """从 Excel 文件中读取 True Label 和 Prediction Probability"""
    data = pd.read_excel(file_path)

    # 检查列是否存在
    if "True Label" not in data.columns or "Prediction Probability" not in data.columns:
        raise KeyError(f"Column 'True Label' or 'Prediction Probability' not found in file {file_path}.")

    y_label = data["True Label"]
    y_scores = data["Prediction Probability"]

    # 使用正则表达式提取概率值
    try:
        y_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in y_scores]
    except AttributeError:
        raise ValueError(f"Prediction Probability format error in file {file_path}.")

    y_scores = np.array(y_scores, dtype=float)
    y_pred_score = 1 - y_scores  # 反转概率值
    return y_label, y_pred_score

def plot_roc_curve(y_true, y_pred_prob, ax, label=None):
    """绘制 ROC 曲线并计算 AUC 值"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')  # 标注 AUC 值
    return ax

if __name__ == '__main__':
    # 文件夹路径
    directory = 'D:\\MONAI-dev\\K_repeats\\val_records\\'
    #directory = "E:\\Ultrasound_data\\much\\"

    # 获取所有 Excel 文件路径
    file_paths = read_excel_files(directory)

    # 创建画布
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 遍历每个文件，绘制 ROC 曲线，并修改图注为 fold1, fold2, fold3...
    for idx, file_path in enumerate(file_paths):
        try:
            y_true, y_pred_prob = read_data(file_path)
            fold_label = f'fold{idx + 1}'  # 图注修改为 fold1, fold2, fold3...
            ax = plot_roc_curve(y_true, y_pred_prob, ax, label=fold_label)
            print(f"Processed file: {file_path} as {fold_label}")
        except (KeyError, ValueError) as e:
            print(f"Skipping file {file_path} due to error: {e}")

    # 绘制随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    # 美化图像
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for 5-Fold Cross Validation')
    ax.legend(loc='lower right')
    ax.grid(True)

    # 显示图像
    plt.tight_layout()
    output_path = 'much_less_plot.png'  # 保存路径
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
