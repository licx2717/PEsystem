import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

# 读取数据
result_1 = pd.read_excel("E:\\Ultrasound_data\\reselected\\test_dataset\\GUexperiment_predictions.xlsx")
result_2 = pd.read_excel("E:\\Ultrasound_data\\reselected\\test_dataset\\model38_predictions.xlsx")

# 提取预测概率
prediction_1 = result_1["Prediction Probability[N/P]"]
prediction_2 = result_2["Prediction Probability[N/P]"]

# 处理预测概率
y1_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in prediction_1]
y1_scores = np.array(y1_scores)
y1_scores = [float(score) if isinstance(score, np.str_) else score for score in y1_scores]

y2_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in prediction_2]
y2_scores = np.array(y2_scores)
y2_scores = [float(score) if isinstance(score, np.str_) else score for score in y2_scores]

# 设定阈值进行二分类
threshold = 0.5
y1_labels = ['P' if score > threshold else 'N' for score in y1_scores]
y2_labels = ['P' if score > threshold else 'N' for score in y2_scores]

# 计算 Kappa 系数
kappa = cohen_kappa_score(y1_labels, y2_labels)

print(f"Kappa coefficient: {kappa}")

# 绘制直方图叠加
plt.figure(figsize=(10, 6))

# 使用 seaborn 绘制直方图
sns.histplot(y1_scores, bins=20, kde=True, color='blue', label='Model 1', alpha=0.5)
sns.histplot(y2_scores, bins=20, kde=True, color='green', label='Model 2', alpha=0.5)

# 设置图表标题和标签
plt.title('Prediction Probability Distributions')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()

# 显示 Kappa 系数
plt.text(0.6, 0.8, f'Kappa Coefficient: {kappa:.2f}', transform=plt.gca().transAxes, fontsize=12)

# 显示图表
plt.show()
