import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample


def bootstrap_calibration_curve(df, n_splits=10, n_bins=10, sample_size=60):
    # 如果未指定样本大小，则默认使用原始数据集的大小
    if sample_size is None:
        sample_size = len(df)

    # 初始化空列表来存储每次bootstrap的结果
    bootstrap_predicted_probs = []
    bootstrap_actual_frequencies = []

    # Bootstrap过程
    for _ in range(n_splits):
        # 有放回抽样，抽取指定数量的样本
        df_resampled = resample(df, replace=True, n_samples=sample_size)

        # 计算校准曲线
        fractions_of_positives, mean_predicted_values = calibration_curve(
            df_resampled['label'],
            df_resampled['trimmed_max_cal'],
            n_bins=n_bins
        )

        # 存储结果
        bootstrap_predicted_probs.append(mean_predicted_values)
        bootstrap_actual_frequencies.append(fractions_of_positives)

    # 转换为numpy数组
    bootstrap_predicted_probs = np.array(bootstrap_predicted_probs)
    bootstrap_actual_frequencies = np.array(bootstrap_actual_frequencies)

    # 计算平均值
    mean_predicted_probs = np.mean(bootstrap_predicted_probs, axis=0)
    mean_actual_frequencies = np.mean(bootstrap_actual_frequencies, axis=0)

    # 计算标准误差
    std_err = np.std(bootstrap_actual_frequencies, axis=0)

    # 计算置信区间
    lower_bound = mean_actual_frequencies - 1.96 * std_err
    upper_bound = mean_actual_frequencies + 1.96 * std_err

    return mean_actual_frequencies, mean_predicted_probs, lower_bound, upper_bound


merge = 'E:\\DoctorProject\\golou\\gulou_final_0\\res\merge\\merge_final_Best_Model_After100_result.csv'
merge = pd.read_csv(merge)

y_merge = merge['label']
y_pred_merge = merge['trimmed_max_cal']

bin = 10

fraction_of_positives_max, mean_predicted_value_max = calibration_curve(y_merge, y_pred_merge, n_bins=bin)



brier_score_max = brier_score_loss(y_merge, y_pred_merge)


print('brier_score_max:', brier_score_max)


# plt.figure(figsize=(10, 6))
#
# plt.plot(fraction_of_positives_max, mean_predicted_value_max, marker='o', linestyle='-', label='y_pred_max')
#
#
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
#
# # 设置图表标题和坐标轴标签
# plt.title('Calibration Curve for y_pred_0')
# plt.xlabel('Mean Predicted Probability')
# plt.ylabel('Fraction of Positives')
# plt.legend(loc="lower right")
#
# plt.show()
'''计算置信区间'''
mean_freq, mean_probs, lower, upper = bootstrap_calibration_curve(merge)

plt.figure()
plt.errorbar(mean_probs, mean_freq, yerr=[mean_freq-lower, upper-mean_freq], fmt='o-', ecolor='black', lw=0.5, capsize=5, label='Calibration with 95% CI')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.title('Calibration Curve with Confidence Interval')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend(loc="lower right")
plt.show()
