import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set default font sizes
plt.rcParams.update(
    {'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

# 目录路径
dir_name = r"E:\\Ultrasound_data\\much\\"


def read_excel_files(directory):
    data_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory, filename)
            data_paths.append(file_path)
    return data_paths


def load_data_and_calculate_auc(filename):
    data = pd.read_excel(filename)
    y_true = data["True Label"]
    y_scores = data["Predicted Probability"]
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr, auc


# 读取Excel文件
figures = read_excel_files(dir_name)

# 绘制ROC曲线
fig, ax = plt.subplots(figsize=(8, 6))

# Colors and labels for small (orange) and large (blue) effusion
colors = ['tab:orange', 'tab:blue'] * 5  # Alternating colors for folds
effusion_types = ['Small effusion', 'Large effusion']  # Legend labels

# Plot ROC curves for each fold
for index, filename in enumerate(figures):
    fpr, tpr, auc = load_data_and_calculate_auc(filename)
    color = colors[index]
    label_type = effusion_types[index % 2]  # Alternates between small and large
    ax.plot(fpr, tpr, color=color, linewidth=2)

# Custom legend entries
custom_lines = [
    plt.Line2D([0], [0], color='tab:orange', lw=2),
    plt.Line2D([0], [0], color='tab:blue', lw=2),
]

# Set title and labels with larger font
ax.set_title('Receiver Operating Characteristic', fontsize=14, pad=20)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)

# Add legend with effusion type info
ax.legend(custom_lines, ['Small effusion', 'Large effusion'],
          loc='lower right', title="Effusion Type:", title_fontsize=11)

# Add grid
ax.grid(True, alpha=0.3)

# Display plot
plt.tight_layout()
plt.savefig('ROC_Curve_300dpi.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()


# Define function to calculate all metrics
def load_data_and_calculate_metrics(filename):
    data = pd.read_excel(filename)
    y_true = data["True Label"]
    y_scores = data["Predicted Probability"]
    y_pred = (y_scores >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return auc, accuracy, precision, recall, f1, specificity


# Store all fold results
results = []
for index, filename in enumerate(figures):
    auc, accuracy, precision, recall, f1, specificity = load_data_and_calculate_metrics(filename)
    results.append({
        'Fold': index + 1,
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'Effusion Type': 'Small' if (index % 2 == 0) else 'Large'
    })

# Prepare radar chart data
categories = ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity']
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Create radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Configure radar chart
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)

# Draw axis lines for each angle
ax.set_rmin(0)
ax.set_rmax(1)

# Group data by effusion type
small_effusion_values = []
large_effusion_values = []

for result in results:
    values = [result['AUC'], result['Accuracy'], result['Precision'],
              result['Recall'], result['Specificity']]
    if result['Effusion Type'] == 'Small':
        small_effusion_values.append(values + values[:1])
    else:
        large_effusion_values.append(values + values[:1])

# Calculate average values for each effusion type
small_avg = np.mean(small_effusion_values, axis=0)
large_avg = np.mean(large_effusion_values, axis=0)

# Plot average values for each effusion type
ax.plot(angles, small_avg, color='tab:orange', linewidth=2, linestyle='solid', label='Small effusion')
ax.fill(angles, small_avg, color='tab:orange', alpha=0.1)

ax.plot(angles, large_avg, color='tab:blue', linewidth=2, linestyle='solid', label='Large effusion')
ax.fill(angles, large_avg, color='tab:blue', alpha=0.1)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Set title with larger font
plt.title('Performance Metrics by Effusion Type', fontsize=14, pad=20)

# Adjust layout
plt.tight_layout()
plt.savefig('Radar_Chart_300dpi.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()

# Print results in tabular format
print("\nPerformance Metrics Summary:")
print("-" * 85)
print(f"{'Fold':<6}{'Type':<12}{'AUC':<8}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}{'Specificity':<12}")
print("-" * 85)
for result in results:
    print(f"{result['Fold']:<6}{result['Effusion Type']:<12}{result['AUC']:<8.2f}{result['Accuracy']:<10.2f}"
          f"{result['Precision']:<10.2f}{result['Recall']:<10.2f}{result['F1 Score']:<10.2f}{result['Specificity']:<12.2f}")
