import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import re
import os
import pandas as pd


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

    y_label = data["True Label"].values
    y_pred_score = data["Prediction Probability"].values
    return y_label, y_pred_score


def plot_roc_with_confidence(y_true, y_scores, set_name, ax, n_bootstrap=500, alpha=0.05):
    """
    绘制 ROC 曲线并添加 95% 置信区间阴影
    Args:
        y_true (np.ndarray): 真实标签（0 或 1）
        y_scores (np.ndarray): 模型输出的分数或概率
        set_name (str): 数据集名称（训练集、验证集或测试集）
        ax (matplotlib.axes.Axes): Matplotlib 的 axes 对象，用于绘制图形
        n_bootstrap (int): 自助采样次数，用于计算置信区间
        alpha (float): 置信水平（默认为 0.05，即 95% 置信区间）
    Returns:
        dict: 包含该数据集的AUC值和置信区间信息
    """
    # 初始化存储所有 bootstrap 的 ROC 曲线和 AUC 值
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 计算真实 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 自助采样计算置信区间
    np.random.seed(12)  # 设置随机种子以保证可重复性
    for _ in range(n_bootstrap):
        # 自助采样
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # 如果采样后只有一个类别，跳过

        # 计算当前采样的 ROC 曲线
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_scores[indices])
        tprs.append(np.interp(mean_fpr, fpr_boot, tpr_boot))
        tprs[-1][0] = 0.0  # 确保曲线从 (0,0) 开始
        aucs.append(auc(fpr_boot, tpr_boot))

    # 计算平均 ROC 曲线和置信区间
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # 确保曲线结束于 (1,1)
    std_tpr = np.std(tprs, axis=0)

    # 计算 95% 置信区间
    auc_ci_lower = np.percentile(aucs, (alpha / 2) * 100)
    auc_ci_upper = np.percentile(aucs, (1 - alpha / 2) * 100)
    tpr_upper = np.minimum(mean_tpr + norm.ppf(1 - alpha / 2) * std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - norm.ppf(1 - alpha / 2) * std_tpr, 0)

    # 绘制 ROC 曲线
    line, = ax.plot(fpr, tpr, lw=2)
    #ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.3, color=line.get_color())

    # 返回AUC信息和置信区间
    return {
        'set_name': set_name,
        'roc_auc': roc_auc,
        'auc_ci_lower': auc_ci_lower,
        'auc_ci_upper': auc_ci_upper,
        'color': line.get_color()
    }


def process_excel_files(directory):
    """处理文件夹中的所有 Excel 文件，绘制 ROC 曲线"""
    file_paths = read_excel_files(directory)

    # 创建一个图形和 axes 对象
    fig, ax = plt.subplots(figsize=(12, 8))

    # 存储所有结果
    results = []

    for file_path in file_paths:
        try:
            # 从 Excel 文件中读取数据
            y_true, y_scores = read_data(file_path)

            # 根据文件名确定数据集类型
            if "train" in os.path.basename(file_path).lower():
                set_name = "Training Set"
                order = 0
            elif "val" in os.path.basename(file_path).lower():
                set_name = "Validation Set"
                order = 1
            elif "test" in os.path.basename(file_path).lower():
                set_name = "Test Set"
                order = 2
            else:
                set_name = "Unknown Set"
                order = 3

            # 绘制 ROC 曲线并获取结果
            result = plot_roc_with_confidence(y_true, y_scores, set_name, ax)
            result['order'] = order
            results.append(result)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # 按照顺序排序: Training, Validation, Test
    results.sort(key=lambda x: x['order'])

    # 自定义图例文本
    legend_handles = []
    legend_labels = []
    for result in results:
        # 格式化为两位小数
        auc_value = f"{result['roc_auc']:.2f}"
        ci_lower = f"{result['auc_ci_lower']:.2f}"
        ci_upper = f"{result['auc_ci_upper']:.2f}"

        # 创建自定义图例条目
        line = plt.Line2D([0], [0], color=result['color'], lw=2)
        legend_handles.append(line)
        legend_labels.append(f"{result['set_name']} (AUC = {auc_value}, 95% CI: {ci_lower}-{ci_upper})")

    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    legend_handles.append(plt.Line2D([0], [0], color='k', linestyle='--', lw=2))
    legend_labels.append('Random (AUC = 0.50)')

    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    # 添加格子线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 美化图例
    legend = ax.legend(legend_handles, legend_labels, loc="lower right",
                       fontsize=10, frameon=True, edgecolor='black')

    plt.savefig('D:\\MONAI-dev\\K_repeats\\val_records\\roc_curves(0.85).png',
                dpi=600, bbox_inches='tight')

    # 显示图形
    plt.show()


# 主程序
directory = 'D:\\MONAI-dev\\K_repeats\\image_data\\'  # Excel 文件所在文件夹路径
data_dir=process_excel_files(directory)

# 运行分析


# 自定义绘图样式（可选）
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.savefig(f"{directory}/roc_curves（mark）.png", dpi=600, bbox_inches='tight')
