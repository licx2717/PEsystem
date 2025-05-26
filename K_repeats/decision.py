import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


file_path="D:\\MONAI-dev\\K_repeats\\val_records\\folds1_predictions.xlsx"
data=pd.read_excel(file_path)
y_true = data["True Label"]
y_scores = data["Prediction Probability[N/P]"]
# 使用列表推导式处理 y_scores 中的每个元素
y_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in y_scores]
y_scores = np.array(y_scores)
y_scores = y_scores.astype(float)

def decision_curve(y_true, y_prob, threshold_values=None):
    """
    Calculate the points for a clinical decision curve.

    Parameters:
    - y_true: True binary labels (0 or 1).
    - y_prob: Predicted probabilities.
    - threshold_values: Threshold values at which to evaluate the curve. If None, defaults to np.linspace(0, 1, 101).

    Returns:
    - thresholds: Thresholds used for evaluation.
    - net_benefit: Net benefit at each threshold.
    """
    if threshold_values is None:
        threshold_values = np.linspace(0, 1, 101)

    # Calculate the number of positive cases
    num_positives = np.sum(y_true == 1)
    num_negatives = len(y_true) - num_positives

    # Initialize arrays to store results
    net_benefit = np.zeros_like(threshold_values)

    # Calculate net benefit for each threshold
    for i, threshold in enumerate(threshold_values):
        # Predictions based on threshold
        predictions = (y_prob >= threshold).astype(int)

        # Calculate true positives, false positives, etc.
        tp = np.sum(predictions * y_true)
        fp = np.sum(predictions) - tp

        # Net benefit formula
        net_benefit[i] = (tp / num_positives) - (fp / num_negatives) * (threshold / (1 - threshold))

    return threshold_values, net_benefit


# Example usage

thresholds, net_benefit = decision_curve(y_true, y_scores)

# Plot the decision curve
plt.figure(figsize=(10, 6))
plt.plot(thresholds, net_benefit, label='Model')
plt.plot(thresholds, thresholds, linestyle='--', color='gray', label='Treat All')
plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', color='black', label='Treat None')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Clinical Decision Curve')
plt.legend()
plt.grid(True)
plt.show()