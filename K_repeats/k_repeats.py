import os
import torch
import monai
import torch.nn as nn
import torch.optim as optim
from monai.metrics import ROCAUCMetric
from sklearn.model_selection import KFold
from dataload.utils import read_split_data
from dataload.my_dataset import MyDataSet
from ResNet.resnet_models import resnet34
from monai.data import decollate_batch
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (Activations, AsDiscrete,RandFlip,AddChannel,RandRotate)
def main():
    # 设置cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 设置路径
    train_root ="D:\\ultrasound_re\\dataset\\train_data"
    Train_images_path, Train_images_label, val_images_path, val_images_label = read_split_data(train_root, val_rate=0)

    # 设置transforms
    data_transform = {
        "train": monai.transforms.Compose([AddChannel(),
                                           RandRotate(prob=1.0),
                                           RandFlip(prob=0.5)
                                           ]),
        "val": monai.transforms.Compose([AddChannel()
                                         ])}
    #--------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=False)
    fold = 1

    for train_index, val_index in kf.split(Train_images_path):
        train_images= [Train_images_path[i] for i in train_index]
        train_label= [Train_images_label[i] for i in train_index]
        val_images = [Train_images_path[i] for i in val_index]
        val_label = [Train_images_label[i] for i in val_index]
    #----------------------------------------------------------------------
        train_data_set = MyDataSet(images_path=train_images,
                                   images_class=train_label,
                                   transform=data_transform["train"])
        train_num = len(train_data_set)
        batch_size = 2
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = monai.data.DataLoader(train_data_set,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=nw, drop_last=True)
        # 设置验证集dataset和dataloader
        validate_data_set = MyDataSet(images_path=val_images,
                                      images_class=val_label,
                                      transform=data_transform["val"])
        val_num = len(validate_data_set)
        validate_loader = monai.data.DataLoader(validate_data_set,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=nw, drop_last=True)
        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))
        post_pred = monai.transforms.Compose([Activations(softmax=True)])
        post_label = monai.transforms.Compose([AsDiscrete(to_onehot=2)])
        # --------------------------------------------------------------------------------------------
        model_name = "resnet34_3d"
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epochs = 4
        save_path="D:\\MONAI-dev\\Kmodels"
        auc_metric = ROCAUCMetric()
        step_lr_list = []
        #_____________________________________________________________
        weights = torch.load("D:\\MONAI-dev\\pretrain\\resnet_34_23dataset.pth")# 设置网络模型 加载预训练权重
        weights = weights['state_dict']
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        net = resnet34(num_classes=2)
        model_dict = net.state_dict()
        model_dict.update(weights)
        net.load_state_dict(model_dict)
        net.to(device)
        # -----------------------------------------------------------------------------
        # loss和optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)
        step_schedule = optim.lr_scheduler.StepLR(step_size=200, gamma=0.1, optimizer=optimizer)  ### 64*epoch
        # 添加tensorboard
        logdir = "D:/MONAI-dev/logs/logs5_fold{}".format(fold)
        os.makedirs(logdir, exist_ok=True)
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
                step = step + 1
                images, labels = batch_data[0].float().to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
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
            epoch_loss = epoch_loss / step
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

                    loss_val = loss_val / len(validate_loader)
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
                        fold_model_path = f"model_Krepeats_dict{epoch+1}.pth"
                        torch.save(net.state_dict(), os.path.join(save_path, fold_model_path))
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

        fold += 1


if __name__ == '__main__':
    main()