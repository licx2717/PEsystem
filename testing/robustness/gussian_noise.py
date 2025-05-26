import os
import torch
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from testing.data_process import read_data
from testing.resnet_models import resnet34
from testing.my_dataset import MyDataSet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testing.add import AddChannel
import numpy as np


def add_gaussian_noise(image, mean=0, std=1):
    """
    添加高斯噪声
    """
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # 确保值在 [0, 1] 范围内
    #print("Pepper shape:", noisy_image.shape)
    return noisy_image


def evaluate_noise_auc(model, test_loader, device, noise_type='gaussian', noise_param=0.1):
    """
    评估模型在噪声/伪影下的 AUC
    """
    model.eval()
    y_true = []  # 原始标签
    y_scores = []  # 对抗样本的预测概率

    for batch in test_loader:
        images, labels = batch[0].float().to(device), batch[1].to(device)
        #print(f"Batch images shape: {images.shape}")  # 打印图像形状
        if noise_type == 'gaussian':
            noisy_images = add_gaussian_noise(images, std=noise_param)
        else:
            raise ValueError("Unknown noise type")

        outputs = model(noisy_images)
        probabilities = F.softmax(outputs, dim=1)[:, 1]  # 获取正类概率

        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probabilities.detach().cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    return auc_score



def main():
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # 加载数据
    train_root = "E:\\Ultrasound_data\\FGSM"
    images_path, images_label = read_data(train_root)

    # 设置数据变换
    data_transform = transforms.Compose([transforms.ToTensor(), AddChannel()])
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 创建测试集 DataLoader
    test_data_set = MyDataSet(images_path=images_path,
                              images_class=images_label,
                              transform=data_transform)
    val_num = len(test_data_set)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=nw,
                                              drop_last=True)
    print("Using {} images for testing.".format(val_num))

    # 加载模型
    weights = torch.load("D:\\MONAI-dev\\Kmodels\\model36_classification3d_dict300.pth",
                         map_location=lambda storage, loc: storage.cuda(0))
    weights = {k: v for k, v in weights.items()}
    net = resnet34()
    model_dict = net.state_dict()
    model_dict.update(weights)
    net.load_state_dict(model_dict)
    net.to(device)

    # 定义噪声/伪影类型和参数
    noise_types =['gaussian']
    noise_params ={'gaussian': [0, 0.05, 0.1, 0.15, 0.20,0.25,0.30]}



    # 评估每种噪声/伪影下的 AUC
    for noise_type in noise_types:
        params = noise_params[noise_type]
        auc_scores = []

        for param in params:
            auc_score = evaluate_noise_auc(net, test_loader, device, noise_type=noise_type, noise_param=param)
            auc_scores.append(auc_score)
            print(f"AUC for {noise_type} noise with param={param}: {auc_score:.4f}")

        # 绘制 AUC 随噪声参数的变化曲线
        # 设置图像样式
        plt.figure(figsize=(12, 8))  # 设置图像大小
        plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
        plt.rcParams['font.size'] = 12  # 设置字号
        # 绘制曲线
        plt.plot(params, auc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        # 设置坐标轴标签
        plt.xlabel('Convolution Kernel Size', fontsize=12, fontweight='bold')
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        # 设置标题
        plt.title(f'AUC vs {noise_type.capitalize()} Noise/Artifact', fontsize=14, fontweight='bold')
        # 设置 y 轴范围
        plt.ylim(0.5, 1.0)  # AUC 的合理范围是 [0, 1]，这里设置为 [0.7, 1.0] 以放大细节
        # 网格线
        plt.grid(True, linestyle='--', alpha=0.6)
        # 调整布局
        plt.tight_layout()
        # 保存图像
        plt.savefig('auc_vs_pepper.png', dpi=600, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
        #plt.savefig('auc_vs_noise.svg', bbox_inches='tight')  # 保存为矢量图 SVG 文件
        # 显示图像
        plt.show()


if __name__ == '__main__':
    main()
