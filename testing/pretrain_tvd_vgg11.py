import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim

from dataload.my_dataset import MyDataSet
from dataload.utils import read_split_data, plot_data_loader_image
from model_vgg.model import vgg
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch, DataLoader
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized,
    ScaleIntensityd, Spacingd, RandCropByPosNegLabeld, RandAffined, RandRotated, RandGaussianNoised
)
from monai.metrics import ROCAUCMetric

from vgg11_model.pre_vgg11_model import VGG11Transfer


def main():
    # 设置cuda
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 设置路径
    train_root = "E:\\HeartWound\\project_pretest\\data\\train_data"
    Train_images_path, Train_images_label, val_images_path, val_images_label = read_split_data(train_root, val_rate=0.2, Rotate=False)

    # 设置transforms
    data_transform = {
        "train": torchvision.transforms.Compose([transforms.ToTensor()]),
        "val": torchvision.transforms.Compose([transforms.ToTensor()])}


    # 设置训练集dataset和dataloader
    train_data_set = MyDataSet(images_path=Train_images_path,
                               images_class=Train_images_label,
                               transform=data_transform["train"])
    train_num = len(train_data_set)
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last = True)

    # 设置验证集dataset和dataloader
    validate_data_set = MyDataSet(images_path=val_images_path,
                                  images_class=val_images_label,
                                  transform=data_transform["val"])
    val_num = len(validate_data_set)
    validate_loader = torch.utils.data.DataLoader(validate_data_set,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw, drop_last = True)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    post_pred = monai.transforms.Compose([Activations(softmax=True)])
    post_label = monai.transforms.Compose([AsDiscrete(to_onehot=2)])

    # 设置网络模型  加载预训练权重
    model_name = 'vgg11'
    net = VGG11Transfer(num_classes=2)
    net.to(device)

    # loss和optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=0.07)
    step_schedule = optim.lr_scheduler.StepLR(step_size=250, gamma=0.1, optimizer=optimizer)### 64*epoch

    # 训练超参
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epochs = 200
    lambda_reg = 0.001
    save_path = './{}Net.pth'.format(model_name)
    auc_metric = ROCAUCMetric()
    step_lr_list = []

    # 添加tensorboard
    logdir = 'E:\\HeartWound\\project_pretest\\logs\\logs16_pretrain'
    writer = SummaryWriter(log_dir=logdir)

    # 模型训练和验证
    for epoch in range(epochs):
        # train
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        net.train()
        epoch_loss = 0.0
        step = 0
        for batch_data in train_loader:
            step += 1
            images, labels = batch_data[0].float().to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            l2_regularization = torch.tensor(0., device=device)
            for param in net.parameters():
                l2_regularization += torch.norm(param.to(device), 2)
            loss += lambda_reg * l2_regularization
            loss.backward()
            optimizer.step()
            step_schedule.step()
            step_lr_list.append(step_schedule.get_last_lr()[0])
            # print statistics
            epoch_loss += loss.item()
            epoch_len = len(train_data_set) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        step_schedule.step()
        step_lr_list.append(step_schedule.get_last_lr()[0])
        epoch_loss /= step
        writer.add_scalars('Loss', {"average_train_loss": epoch_loss}, epoch + 1)  # add_scalars可将多个变量放进一个图中
        print(f"epoch {epoch + 1} average_train loss: {epoch_loss:.4f}")
        print('----------current_lr-----------', optimizer.state_dict()['param_groups'][0]['lr'])



        # validate
        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                # train_set#############
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for train_data in train_loader:
                    train_images, train_labels = train_data[0].float().to(device), train_data[1].to(device)
                    y_pred = torch.cat([y_pred, net(train_images)], dim=0)
                    y = torch.cat([y, train_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot

                print(
                    "current epoch: {} train accuracy: {:.4f} train AUC: {:.4f}".format(
                        epoch + 1, acc_metric, auc_result
                    )
                )
                writer.add_scalars('AUC', {"train_auc": auc_result}, epoch + 1)
                writer.add_scalars('Accuracy', {"train_accuracy": acc_metric}, epoch + 1)

                # val_set
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                loss_val = 0
                for val_data in validate_loader:
                    val_images, val_labels = val_data[0].float().to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, net(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    loss_tmp = loss_function(net(val_images), val_labels)
                    loss_val += loss_tmp.item()

                loss_val = loss_val/len(validate_loader)
                writer.add_scalars('Loss', {"average_val_loss": loss_val}, epoch + 1)
                print(f"epoch {epoch + 1} average_val loss: {loss_val:.4f}")
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    #torch.save(net.state_dict(), save_path, "best_metric_model_classification3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} val accuracy: {:.4f} val AUC: {:.4f} best val accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalars('AUC', {"val_auc": auc_result}, epoch + 1)
                writer.add_scalars('Accuracy', {"val_accuracy": acc_metric}, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()



# %%
if __name__ == '__main__':
    main()
