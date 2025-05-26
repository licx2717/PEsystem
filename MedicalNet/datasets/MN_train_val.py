import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models import resnet
import torch.optim as optim
from monai.losses import DiceLoss
import nibabel as nib
from datasets.mydataset import MyDataset
from torch.utils.data import DataLoader
from utils.logger import log
import time
import numpy as np
from scipy import ndimage
import os
#----------------------------------------------------
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = 2.0 * torch.sum(y_true * y_pred) + epsilon
    union = torch.sum(y_true) + torch.sum(y_pred) + epsilon
    return intersection / union
def dice_loss(y_true, y_pred, epsilon=1e-6):
    intersection = 2.0 * torch.sum(y_true * y_pred) + epsilon
    union = torch.sum(y_true) + torch.sum(y_pred) + epsilon
    dice_coeff = intersection / union
    return 1.0 - dice_coeff
#-----------------------------------------------------
def save_prediction_as_nrrd(prediction, filename):
    # 将预测结果转换为numpy数组
    prediction_numpy = prediction.cpu().detach().numpy()
    # 创建一个新的nrrd图像
    nrrd_image = nib.Nifti1Image(prediction_numpy, np.eye(4))
    # 保存nrrd图像
    nib.save(nrrd_image, filename)
#-------------------------------------------------------------------
def main():
    train_list_path = 'E:\\heart_seg\\Datasets\\train_data\\train_seg.txt'
    train_dir = "E:\\heart_seg\\Datasets\\train_data"
    training_dataset = MyDataset(root_dir=train_dir, img_list=train_list_path, input_D=32, input_H=224, input_W=224,
                                 phase='train')
    traindata_loader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)

    val_list_path = 'E:\\heart_seg\\Datasets\\val_data\\val_seg.txt'
    val_dir = "E:\\heart_seg\\Datasets\\val_data"
    val_dataset = MyDataset(root_dir=val_dir, img_list=val_list_path, input_D=32, input_H=224, input_W=224,
                                 phase='val')
    valdata_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)

    model = resnet.resnet50(
        sample_input_W=224,
        sample_input_H=224,
        sample_input_D=32,
        shortcut_type='B',
        no_cuda=False,
        num_seg_classes=1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[1])
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(
        'E:\\heart_seg\\Segmentation\\pretrained\\resnet_50_23dataset.pth'))
    pretrain = torch.load('E:\\heart_seg\\Segmentation\\pretrained\\resnet_50_23dataset.pth')
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict)  # 字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict)  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    for pname, p in model.named_parameters():  # 返回各层中参数名称和数据。
        for layer_name in ['conv_seg']:
            if pname.find(layer_name) >= 0:
                print(pname)
    new_parameters = []
    for pname, p in model.named_parameters():  # 返回各层中参数名称和数据。
        for layer_name in ['conv_seg']:
            if pname.find(layer_name) >= 0:
                new_parameters.append(p)
                break

    new_parameters_id = list(map(id, new_parameters))
    base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    parameters = {'base_parameters': base_parameters,
                  'new_parameters': new_parameters}
    learning_rate = 5e-4
    params = [
        {'params': parameters['base_parameters'], 'lr': learning_rate},
        {'params': parameters['new_parameters'], 'lr': learning_rate * 10}
    ]
    #optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-2)
    optimizer = optim.Adam(params, weight_decay=1e-3)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = optim.lr_scheduler.StepLR(step_size=100, gamma=0.5, optimizer=optimizer)

    total_epochs = 500
    batches_per_epoch = len(traindata_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    #loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_seg = DiceLoss(sigmoid=True, squared_pred=False)

    writer = SummaryWriter('E:\\heart_seg\\Segmentation\\Seg_logs\\MNseg_logs9')
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        log.info('Start epoch {}'.format(epoch))
        log.info('lr = {}'.format(scheduler.get_last_lr()))
        for batch_id, batch_data in enumerate(traindata_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data
            volumes = volumes.cuda()
            optimizer.zero_grad()
            out_masks = model(volumes)
            print(out_masks.shape)
            # resize label 如果大小不相同，进行缩放
            [n, _, d, h, w] = out_masks.shape  # n = batch_size
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                 label_mask = label_masks[label_id]
                 [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                 label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                 scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                 label_mask = ndimage.zoom(label_mask, scale, order=0)
                 new_label_masks[label_id] = label_mask
#----------------------------------------------------------------------------------------------
            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            new_label_masks=new_label_masks.unsqueeze(1)
            new_label_masks = new_label_masks.cuda()
            new_label_masks = new_label_masks.float()
            new_label_masks = new_label_masks.to(device)
            # calculating loss
            loss_value_seg = loss_seg(out_masks, new_label_masks)
            #out_masks = torch.argmax(out_masks, dim=1)
            #dice = dice_coef(out_masks, new_label_masks)
            dice = 1-loss_value_seg
            loss = loss_value_seg
            loss.backward()
            optimizer.step()
            scheduler.step()
            # -------------------------
            epoch_loss += loss.item() * volumes.size(0)
            epoch_dice += dice.item() * volumes.size(0)

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
        epoch_loss = epoch_loss / len(traindata_loader.dataset)
        epoch_dice = epoch_dice / len(traindata_loader.dataset)
        writer.add_scalars('LOSS', {"train_loss": epoch_loss}, epoch + 1)
        writer.add_scalars('DICE', {"train_dice": epoch_dice}, epoch + 1)


        # 打印平均损失和Dice系数
        print('Epoch {}/{} Train_Loss: {:.4f} Train_Dice: {:.4f}'.format(epoch+1, total_epochs, epoch_loss, epoch_dice))

        model.eval()  # switch model to evaluation mode
        total_val_loss = 0
        total_val_dice = 0
        val_per_epoch = len(valdata_loader)
        with torch.no_grad():
            for batch_id, val_batch_data in enumerate(valdata_loader):
                # getting data batch
                batch_id_sp = epoch * val_per_epoch
                val_volumes, val_label_masks = val_batch_data
                val_volumes = val_volumes.cuda()
                val_out_masks = model(val_volumes)
                # print(out_masks.shape)
                # resize label 如果大小不相同，进行缩放
                [n, _, d, h, w] =val_out_masks.shape  # n = batch_size
                new_val_masks = np.zeros([n, d, h, w])
                for label_id in range(n):
                    val_label_mask = val_label_masks[label_id]
                    [ori_c, ori_d, ori_h, ori_w] = val_label_mask.shape
                    val_label_mask = np.reshape(val_label_mask, [ori_d, ori_h, ori_w])
                    scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                    val_label_mask = ndimage.zoom(val_label_mask, scale, order=0)
                    new_val_masks[label_id] = val_label_mask
                # ----------------------------------------------------------------------------------------------
                new_val_masks = torch.tensor(new_val_masks).to(torch.int64)
                new_val_masks = new_val_masks.unsqueeze(1)
                new_val_masks = new_val_masks.cuda()
                new_val_masks = new_val_masks.float()
                new_val_masks = new_val_masks.to(device)

                # calculating loss
                val_loss = loss_seg(val_out_masks, new_val_masks)
                # out_masks = torch.argmax(out_masks, dim=1)
                #val_dice = dice_coef(val_out_masks, new_val_masks)
                val_dice = 1 - val_loss
                # -------------------------
                total_val_loss += val_loss.item() * volumes.size(0)
                total_val_dice += val_dice.item() * volumes.size(0)
            val_loss = total_val_loss / len(valdata_loader.dataset)
            val_dice = total_val_dice / len(valdata_loader.dataset)

            writer.add_scalars('LOSS', {"val_loss": val_loss}, epoch + 1)
            writer.add_scalars('DICE', {"val_dice": val_dice}, epoch + 1)

            # 打印平均损失和Dice系数
            print('Epoch {}/{} Val_Loss: {:.4f} Val_Dice: {:.4f}'.format(epoch+1, total_epochs, val_loss, val_dice))
        # 在每个epoch结束后，检查epoch的编号
        if epoch+1 % 100 == 0:
            # 如果epoch是100的倍数，就保存模型
            torch.save(model.state_dict(), 'E:/heart_seg/Segmentation/Seg_models/model_epoch_{}.pth'.format(epoch+1))

    print('Finished training')
if __name__ == '__main__':
    main()



