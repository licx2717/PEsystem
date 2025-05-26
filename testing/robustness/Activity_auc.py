import os
import torch
from monai.transforms import Rand3DElastic
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testing.data_process import read_data
from testing.resnet_models import resnet34
from testing.my_dataset import MyDataSet
from testing.add import AddChannel

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_rotation(image, angle):
    """
    模拟角度变化
    Args:
        image (torch.Tensor): 输入图像（C, H, W）
        angle (float): 旋转角度
    Returns:
        torch.Tensor: 旋转后的图像
    """
    return TF.rotate(image, angle)


def add_elastic_deformation(image, alpha, sigma):
    """
    使用 MONAI 的 Rand3DElastic 实现弹性形变
    Args:
        image (torch.Tensor): 输入图像 (C, H, W) 或 (C, D, H, W)
        alpha (float): 形变强度
        sigma (float): 高斯滤波的标准差
    Returns:
        torch.Tensor: 形变后的图像
    """
    # Step 1: 确保输入图像维度正确 [batch, C, D, H, W]
    while image.dim() < 5:
        image = image.unsqueeze(0)  # 补全到 5D [batch, C, D, H, W]

    # Step 2: 提取空间维度 (D, H, W)
    depth, height, width = image.shape[-3:]

    # Step 3: 设置 spatial_size 为 (D, H, W)
    spatial_size = (depth, height, width)

    #print(f"[Debug] Input shape: {image.shape}, spatial_size={spatial_size}")

    # Step 4: 定义 MONAI 的 Rand3DElastic 转换
    transform = Rand3DElastic(
        sigma_range=(sigma, sigma),
        magnitude_range=(alpha, alpha),
        prob=1.0,
        spatial_size=spatial_size,  # 必须为 (D, H, W)
        mode="bilinear",
    )

    # Step 5: 应用形变
    image=image.squeeze(0)
    deformed_image = transform(image)
    #print("shabi:",deformed_image.shape)
    # Step 6: 压缩多余的 batch 维度 (如果有)
    return deformed_image#.unsqueeze(0)  # 恢复到 [C, D, H, W]


def evaluate_noise_auc(model, test_loader, device, noise_type='rotation', noise_param=(10, 3)):
    """
    评估模型在噪声/伪影下的 AUC
    Args:
        noise_type: 噪声类型 ('rotation' 或 'elastic')
        noise_param: (alpha, sigma) 或 angle
    """
    model.eval()
    y_true = []  # 原始标签
    y_scores = []  # 对抗样本的预测概率

    for batch in test_loader:
        images, labels = batch[0].float().to(device), batch[1].to(device)

        if noise_type == 'rotation':
            noisy_images = torch.stack([add_rotation(img, noise_param) for img in images])
        elif noise_type == 'elastic':
            alpha, sigma = noise_param
            noisy_images = torch.stack([add_elastic_deformation(img, alpha, sigma) for img in images])
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
    noise_types =["rotation"]#['elastic']
    noise_params = {
        #'elastic': [(0, 3), (10, 3), (20, 3), (30, 3), (40, 3),(50, 3),(60,3)]  # [(alpha, sigma), ...]
        'rotation': [0, 10, 20, 30, 40, 50, 60]
    }

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
        #plt.plot([p[0] for p in params], auc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        plt.plot(params, auc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

        plt.xlabel('Elastic Deformation Alpha', fontsize=12, fontweight='bold')
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        plt.title('AUC vs Elastic Deformation Noise', fontsize=14, fontweight='bold')
        plt.ylim(0.5, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'auc_vs_{noise_type}.png', dpi=600, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
