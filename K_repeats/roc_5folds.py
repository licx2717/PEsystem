import os
import torch
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from testing.data_process import read_data
from monai.metrics import ROCAUCMetric
from testing.resnet_models import resnet34
from my5folds_dataset import MyDataSet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testing.add import AddChannel
#---------------------------------------------------------------------------------
def read_paths_and_labels(file_path):
    images_path = []  # 存储图片路径
    images_label = []  # 存储图片标签
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                path = line.strip()  # 去除换行符和空白字符
                if os.path.exists(path):  # 检查路径是否存在
                    if "Positive" in path:
                        label = 1  # 如果路径包含 "Positive"，标签为 1
                    elif "Negative" in path:
                        label = 0  # 如果路径包含 "Negative"，标签为 0
                    else:
                        print(f"Unknown folder for path: {path}. Skipping.")
                        continue
                    images_path.append(path)
                    images_label.append(label)
                else:
                    print(f"The path {path} does not exist. Skipping.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return images_path, images_label
def main():
    # --------------------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    post_pred = transforms.Compose([lambda x: F.softmax(x, dim=0)])
    post_label = transforms.Compose([lambda x: F.one_hot(x, num_classes=2)])


    predict_path = "D:\\MONAI-dev\\K_repeats\\val_records\\fold_5_val.txt"
    images_path, images_label = read_paths_and_labels(predict_path)

    # 设置transforms
    data_transform = transforms.Compose([transforms.ToTensor(), AddChannel()])
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 设置验证集dataset和dataloader
    test_data_set = MyDataSet(images_path=images_path,
                              images_class=images_label,
                              transform=data_transform)
    val_num = len(test_data_set)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=nw,
                                              drop_last=True)
    print("using {} images for testing.".format(val_num))



    # 设置网络模型  加载预训练权重
    # ---------------------------------------------------------------------------------------------
    models = []
    weights = torch.load("D:\\MONAI-dev\\Kmodels\\model38_classification3d_dict5.pth",
                         map_location=lambda storage, loc: storage.cuda(0))
    weights = {k: v for k, v in weights.items()}
    net = resnet34()
    # net = resnet.resnet34(num_classes=2)
    model_dict = net.state_dict()
    model_dict.update(weights)
    net.load_state_dict(model_dict)
    net.to(device)
    # -----------------------------------------------------------------------------
    auc_metric = ROCAUCMetric()

    net.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        data_path = []
        for test_data in test_loader:
            test_images, test_labels = test_data[0].float().to(device), test_data[1].to(device)
            y_pred = torch.cat([y_pred, net(test_images)], dim=0)
            y = torch.cat([y, test_labels], dim=0)
            data_path.append(test_images)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in (y)]
        y_pred_act = [post_pred(i) for i in (y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()
        print(f" Accuracy: {acc_metric:.4f} AUC_result: {auc_result}")
        # -----------------------------------------------------------------------
        y_pred_act_np = [i.cpu().numpy() for i in y_pred_act]
        df = pd.DataFrame({"images_path": images_path,
                           'True Label': images_label,
                           'Predicted Label': y_pred.argmax(dim=1).cpu().numpy(),
                           "Prediction Probability": y_pred[:, 1].cpu().numpy()})
        df.to_excel('C:/Users/16921/Desktop/fold5_val_predictions.xlsx', index=False)

        y_true = y.cpu().numpy()
        y_pred_prob = y_pred[:, 1].cpu().numpy()
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Model evaluation')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    main()
