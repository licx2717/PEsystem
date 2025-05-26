import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

# 读取数据
result_1 = pd.read_excel("E:\\Ultrasound_data\\reselected\\test_dataset\\GUexperiment_predictions.xlsx")
result_2 = pd.read_excel("E:\\Ultrasound_data\\reselected\\test_dataset\\model38_predictions.xlsx")

# 提取预测概率
prediction_1 = result_1["Prediction Probability[N/P]"]
prediction_2 = result_2["Prediction Probability[N/P]"]

# 处理预测概率
y1_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in prediction_1]
y1_scores = np.array([float(score) for score in y1_scores])

y2_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in prediction_2]
y2_scores = np.array([float(score) for score in y2_scores])

# 将预测概率转换为二分类预测
y1_pred = (y1_scores > 0.5).astype(int)
y2_pred = (y2_scores > 0.5).astype(int)

# 计算预测结果的相关性
correlation = np.corrcoef(y1_pred, y2_pred)[0, 1]
print(f"预测结果的相关性: {correlation}")

# 计算 Cohen's Kappa 分数
kappa = cohen_kappa_score(y1_pred, y2_pred)
print(f"Cohen's Kappa 分数: {kappa}")

# 评估模型性能
true_labels = result_1['True Labels']  # 假设真实标签在 `True Labels` 列中

accuracy_1 = accuracy_score(true_labels, y1_pred)
f1_1 = f1_score(true_labels, y1_pred, average='weighted')

accuracy_2 = accuracy_score(true_labels, y2_pred)
f1_2 = f1_score(true_labels, y2_pred, average='weighted')

print(f"Model 1 Accuracy: {accuracy_1}, F1 Score: {f1_1}")
print(f"Model 2 Accuracy: {accuracy_2}, F1 Score: {f1_2}")

# 集成模型
ensemble_predictions = (y1_scores + y2_scores) / 2
ensemble_predictions = (ensemble_predictions > 0.5).astype(int)

ensemble_accuracy = accuracy_score(true_labels, ensemble_predictions)
ensemble_f1 = f1_score(true_labels, ensemble_predictions, average='weighted')

print(f"Ensemble Model Accuracy: {ensemble_accuracy}, F1 Score: {ensemble_f1}")

# 可视化预测结果
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y1_scores, y=y2_scores, hue=true_labels, palette='viridis')
plt.xlabel('Model 1 Prediction Scores')
plt.ylabel('Model 2 Prediction Scores')
plt.title('Comparison of Model Predictions')
plt.show()
