import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import re
from scipy.stats import norm, t

# 目录路径
dir_name = "D:\\MONAI-dev\\K_repeats\\val_records"

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

# 设置随机数种子
np.random.seed(42)  # 可以选择任意整数作为种子

# 定义一个函数来读取数据并计算AUC
def load_data_and_calculate_auc(filename):
    # 使用pandas读取Excel文件
    y_true, y_scores = read_data(filename)
    auc = roc_auc_score(y_true, y_scores)
    return auc

# 遍历目录下的所有Excel文件
file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith('.xlsx')]

# 计算每个文件的AUC值
auc_values = []
for filename in file_list:
    try:
        auc = load_data_and_calculate_auc(filename)
        auc_values.append(auc)
        print(f"{filename}: AUC = {auc:.4f}")
    except Exception as e:
        print(f"读取 {filename} 出错: {e}")

# 计算统计量
mean_auc = np.mean(auc_values)
std_auc = np.std(auc_values, ddof=1)  # 样本标准差的无偏估计
n = len(auc_values)

# 对于小样本量(如5-fold)，使用t分布而不是正态分布
t_critical = t.ppf(0.975, df=n-1)  # 95%置信水平对应的t值
margin_of_error = t_critical * (std_auc / np.sqrt(n))
confidence_interval = (mean_auc - margin_of_error, mean_auc + margin_of_error)

print(f"\n统计结果:")
print(f"平均AUC: {mean_auc:.4f}")
print(f"标准差: {std_auc:.4f}")
print(f"95%置信区间: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")

