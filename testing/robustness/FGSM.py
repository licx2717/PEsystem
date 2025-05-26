import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import io
from torchvision import transforms
import torch.nn.functional as F
from testing.data_process import read_data
from monai.metrics import ROCAUCMetric
from testing.resnet_models import resnet34
from testing.my_dataset import MyDataSet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testing.add import AddChannel
import torch.autograd as autograd
import nibabel as nib  # 引入 nibabel 库用于处理 .nii.gz 文件


# 定义 FGSM 攻击方法
def fgsm_attack(model, inputs, labels, epsilon):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed_inputs = inputs + epsilon * inputs.grad.sign()
    perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    return perturbed_inputs.detach()


def generate_adversarial(model, images, labels, epsilon, save_path=None):
    """
    生成对抗样本，并保存扰动后的图像为 .nii.gz 文件
    """
    adv_images = fgsm_attack(model, images, labels, epsilon)
    if save_path:
        # 将张量转换为 NumPy 数组
        adv_images_np = adv_images.squeeze().cpu().numpy()
        # 创建 NIfTI 图像对象
        nii_img = nib.Nifti1Image(adv_images_np, affine=np.eye(4))
        # 保存为 .nii.gz 文件
        nib.save(nii_img, save_path)
    return adv_images


def evaluate_adversarial_auc(model, test_loader, device, epsilon=0.1):
    """
    评估对抗样本的 AUC
    """
    model.eval()
    y_true = []  # 原始标签
    y_scores = []  # 对抗样本的预测概率

    for batch in test_loader:
        images, labels = batch[0].float().to(device), batch[1].to(device)
        adv_images = generate_adversarial(model, images, labels, epsilon)
        outputs = model(adv_images)
        probabilities = F.softmax(outputs, dim=1)[:, 1]  # 获取正类概率

        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probabilities.detach().cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    return auc_score


def calculate_std(test_loader, device):
    """
    计算数据集的标准差
    """
    all_images = []
    for batch in test_loader:
        images = batch[0].float().to(device)
        all_images.append(images)
    # 将所有图像拼接为一个张量
    all_images = torch.cat(all_images, dim=0)
    # 计算全局标准差
    std = torch.std(all_images).item()
    return std


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

    # 计算数据集的标准差
    std = calculate_std(test_loader, device)
    print(f"Standard deviation of the dataset: {std:.6f}")

    # 基于标准差设置干扰强度
    k_values = [0, 0.005, 0.01, 0.015, 0.020, 0.025, 0.030]  # 标准差的倍数
    epsilons = [k * std for k in k_values]
    print("Epsilon values based on std:", epsilons)

    auc_scores = []

    # 计算每个干扰强度下的 AUC，并保存扰动后的图像
    for idx, epsilon in enumerate(epsilons):
        # 生成对抗样本并保存
        save_path = f"adv_image_eps{idx}.nii.gz"
        auc_score = evaluate_adversarial_auc(net, test_loader, device, epsilon)
        auc_scores.append(auc_score)
        print(f"AUC at ε={epsilon:.6f}: {auc_score:.4f}")

        # 保存扰动后的图像
        for batch in test_loader:
            images, labels = batch[0].float().to(device), batch[1].to(device)
            generate_adversarial(net, images, labels, epsilon, save_path=save_path)
            break  # 只保存一个样本

    # 绘制 AUC 随干扰强度的变化曲线
    plt.figure(figsize=(12, 8))  # 设置图像大小
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
    plt.rcParams['font.size'] = 12  # 设置字号
    plt.plot(epsilons, auc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    plt.ylabel('AUC', fontsize=12, fontweight='bold')
    plt.title('AUC vs Adversarial Perturbation (ε)', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 0.9)  # 设置 y 轴范围
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # 保存图像为 TIFF 格式
    buf = io.BytesIO()
    plt.savefig(buf, format='tiff', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img.save('auc_vs_epsilon.tiff', format='TIFF', dpi=(300, 300))
    buf.close()
    plt.show()


if __name__ == '__main__':
    main()
