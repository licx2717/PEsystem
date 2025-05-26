import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from testing.data_process import read_data
from testing.resnet_models import resnet34
from testing.my_dataset import MyDataSet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testing.add import AddChannel
import numpy as np
import cv2

def add_blur(image, kernel_size=(5, 5), sigma=1.0):
    """
    给图像添加模糊效果
    Args:
        image (np.ndarray): 输入图像
        kernel_size (tuple): 高斯核大小，例如 (5, 5)
        sigma (float): 高斯核的标准差
    Returns:
        np.ndarray: 模糊后的图像
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    # 将值裁剪到 [0, 1] 范围（假设图像值在 [0, 1] 范围内）
    blurred_image = np.clip(blurred_image, 0, 1)
    return blurred_image
def add_jitter(image, jitter_range=0.05):
    """
    给图像添加抖动效果
    Args:
        image (np.ndarray): 输入图像
        jitter_range (float): 抖动的最大范围（相对于图像值），例如 0.05 表示 ±5% 的抖动
    Returns:
        np.ndarray: 抖动后的图像
    """
    jitter = np.random.uniform(low=-jitter_range, high=jitter_range, size=image.shape)
    jittered_image = image + jitter
    # 将值裁剪到 [0, 1] 范围（假设图像值在 [0, 1] 范围内）
    jittered_image = np.clip(jittered_image, 0, 1)
    return jittered_image




def add_blur_and_jitter(image, kernel_size=(5, 5), sigma=1.0, jitter_range=0.05):
    """
    给图像添加模糊和抖动效果
    Args:
        image (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)
        kernel_size (tuple): 高斯核大小，例如 (5, 5)
        sigma (float): 高斯核的标准差
        jitter_range (float): 抖动的最大范围（相对于图像值），例如 0.05 表示 ±5% 的抖动
    Returns:
        torch.Tensor: 添加模糊和抖动后的图像
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (batch_size, height, width, channels)

    # 逐张图像添加模糊和抖动
    noisy_images = []
    for i in range(image_np.shape[0]):
        noisy_image = add_blur(image_np[i], kernel_size, sigma)
        noisy_image = add_jitter(noisy_image, jitter_range)
        noisy_images.append(noisy_image)

    # 将 NumPy 数组转换回 PyTorch 张量
    noisy_images = np.stack(noisy_images, axis=0)  # (batch_size, height, width, channels)
    noisy_images = torch.from_numpy(noisy_images).permute(0, 3, 1, 2).to(image.device)  # (batch_size, channels, height, width)

    return noisy_images


def evaluate_noise_auc(model, test_loader, device, noise_type='gaussian', noise_param=0.1):
    """
    评估模型在噪声/伪影下的 AUC
    """
    model.eval()
    y_true = []  # 原始标签
    y_scores = []  # 对抗样本的预测概率

    for batch in test_loader:
        images, labels = batch[0].float().to(device), batch[1].to(device)

        if noise_type == 'blur':
            # 添加模糊和抖动
            if isinstance(noise_param, int):  # 如果 kernel_size 是整数，转换为元组
                kernel_size = (noise_param, noise_param)
            else:
                kernel_size = noise_param
            noisy_images = add_blur_and_jitter(images, kernel_size=kernel_size)
        else:
            raise ValueError("Unknown noise type")

        with torch.no_grad():
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
    noise_types = ['blur']
    noise_params = {'blur': [1, 2, 3, 4, 5]}  # 模糊的卷积核大小

    # 评估每种噪声/伪影下的 AUC
    for noise_type in noise_types:
        params = noise_params[noise_type]
        auc_scores = []

        for param in params:
            auc_score = evaluate_noise_auc(net, test_loader, device, noise_type=noise_type, noise_param=param)
            auc_scores.append(auc_score)
            print(f"AUC for {noise_type} noise with param={param}: {auc_score:.4f}")

        # 绘制 AUC 随噪声参数的变化曲线
        plt.figure(figsize=(12, 8))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.plot(params, auc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        plt.xlabel('Convolution Kernel Size', fontsize=12, fontweight='bold')
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        plt.title(f'AUC vs {noise_type.capitalize()} Noise/Artifact', fontsize=14, fontweight='bold')
        plt.ylim(0.5, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('auc_vs_blur.png', dpi=600, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
