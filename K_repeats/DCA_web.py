import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from sklearn.metrics import confusion_matrix


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):  #Plot
    ax.plot(thresh_group, net_benefit_model,color = 'crimson', label = 'Model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')
    return ax


def read_excel_files(directory):
    data_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory, filename)
            data_paths.append(file_path)
    return data_paths

def read_data(file_path):
    data = pd.read_excel(file_path)
    y_label = data["True Label"]
    y_scores = data["Prediction Probability[N/P]"]
    # 使用列表推导式处理 y_scores 中的每个元素
    y_scores = [re.search(r'\[(\d+\.\d+)', score).group(1) for score in y_scores]
    y_scores = np.array(y_scores)
    y_pred_score = y_scores.astype(float)
    y_pred_score = 1 - y_pred_score
    return y_label, y_pred_score


if __name__ == '__main__':
    dir_name = 'D:\\MONAI-dev\\K_repeats\\val_records\\'
    file_paths=read_excel_files(dir_name)

    fig, ax = plt.subplots()
    # 遍历每个文件并绘制曲线
    for file in file_paths:
        y_label, y_pred_score = read_data(file)
        thresh_group = np.arange(0, 1, 0.01)
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
        net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
        ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)

    plt.show()
