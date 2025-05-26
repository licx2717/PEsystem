import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models import resnet_conseg
from datasets.mydataset import MyDataset
from torch.utils.data import DataLoader
from utils.logger import log
import scipy.ndimage as ndimage
import time
import torch.optim as optim
import numpy as np


class RobustDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.4, gamma=0.1, eps=1e-6):
        """
        alpha: 体积级损失权重
        beta: 切片级损失权重
        gamma: 连续性约束权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.slice_loss = nn.BCEWithLogitsLoss(reduction='none')  # 用于切片级权重计算

    def _volume_dice(self, pred, target):
        # 标准3D Dice计算
        intersection = 2 * (pred * target).sum() + self.eps
        union = pred.sum() + target.sum() + self.eps
        return 1 - (intersection / union)

    def _slice_level_loss(self, pred, target):
        B, _, D, H, W = pred.shape
        pred = pred.sigmoid()

        # 逐切片计算Dice
        pred_slices = pred.unbind(2)  # 按深度维度解绑 [B,1,H,W]×D
        target_slices = target.unbind(2)

        slice_losses = []
        worst_case_penalty = 0

        for i, (p, t) in enumerate(zip(pred_slices, target_slices)):
            # 基础切片Dice
            intersection = 2 * (p * t).sum((1, 2, 3)) + self.eps
            union = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + self.eps
            dice = intersection / union

            # 动态权重：对低质量切片自动增强权重
            with torch.no_grad():
                slice_quality = 1 - self.slice_loss(p, t).mean((1, 2, 3))  # [B]
                weights = 1.0 / (slice_quality + 0.1)  # 质量越差权重越高

            # 加权后的切片损失
            weighted_loss = weights * (1 - dice)
            slice_losses.append(weighted_loss.mean())

            # 对最差5%切片额外惩罚
            if i % 5 == 0:  # 每隔5片检查一次
                q = torch.quantile(1 - dice, 0.95)
                worst_case_penalty += torch.relu((1 - dice) - q).mean()

        return torch.stack(slice_losses).mean() + 0.1 * worst_case_penalty

    def _spatial_constraint(self, pred, target):
        # 3D梯度惩罚（防止相邻切片突变）
        pred = pred.sigmoid()
        dx = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dy = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        dz = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])

        target_grad = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        grad_penalty = (dx * torch.exp(-5 * target_grad.abs())).mean()
        return grad_penalty

    def forward(self, pred, target):
        vol_loss = self._volume_dice(pred.sigmoid(), target)
        slice_loss = self._slice_level_loss(pred, target)
        spatial_loss = self._spatial_constraint(pred, target)

        total_loss = (
                self.alpha * vol_loss +
                self.beta * slice_loss +
                self.gamma * spatial_loss
        )

        return total_loss


class SliceWiseMonitor:
    """增强的切片级质量监控系统"""

    def __init__(self, num_slices=32):
        self.slice_dice_stats = []
        self.num_slices = num_slices
        # 初始化每个切片的统计信息
        self.slice_stats = {
            f'slice_{i}': {'dices': [], 'losses': []}
            for i in range(num_slices)
        }

    def update(self, pred, target):
        """更新每个切片的性能统计"""
        pred_sig = torch.sigmoid(pred)
        batch_size = pred.shape[0]

        # 按切片计算Dice系数
        for z in range(self.num_slices):
            p_slice = pred_sig[:, :, z, :, :]
            t_slice = target[:, :, z, :, :]

            intersection = 2 * (p_slice * t_slice).sum()
            union = p_slice.sum() + t_slice.sum()
            dice = (intersection / (union + 1e-6)).item()

            self.slice_stats[f'slice_{z}']['dices'].append(dice)

    def get_worst_slices(self, k=3):
        """获取表现最差的k个切片"""
        avg_dices = [
            (i, np.mean(stats['dices']))
            for i, stats in self.slice_stats.items()
        ]
        return sorted(avg_dices, key=lambda x: x[1])[:k]

    def reset_epoch(self):
        """每个epoch开始时重置统计"""
        for stats in self.slice_stats.values():
            stats['dices'] = []
            stats['losses'] = []

#-------------------------------------------------------------------
def main():
    # Initialize dataset and dataloaders (same as before)
    train_list_path = 'E:\\Ultrasound_data\\Segmentation_folder\\datasets\\train_data\\train_seg.txt'
    train_dir = "E:\\Ultrasound_data\\Segmentation_folder\\datasets\\train_data\\"
    training_dataset = MyDataset(root_dir=train_dir, img_list=train_list_path, input_D=32, input_H=224, input_W=224,
                                 phase='train')
    traindata_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True,
                                  drop_last=False)
    val_list_path = 'E:\\Ultrasound_data\\Segmentation_folder\\datasets\\val_data\\val_seg.txt'
    val_dir = "E:\\Ultrasound_data\\Segmentation_folder\\datasets\\val_data\\"
    val_dataset = MyDataset(root_dir=val_dir, img_list=val_list_path, input_D=32, input_H=224, input_W=224,
                            phase='val')
    valdata_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True,
                                drop_last=False)
    # Initialize model (same as before)
    model = resnet_conseg.resnet50(
        sample_input_W=224,
        sample_input_H=224,
        sample_input_D=32,
        shortcut_type='B',
        no_cuda=False,
        num_seg_classes=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0])

    # Load pretrained weights (same as before)
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(
        'D:\\pythonProject1\\MedicalNet-master\\pretrained\\resnet_50_23dataset.pth'))
    pretrain = torch.load('D:\\pythonProject1\\MedicalNet-master\\pretrained\\resnet_50_23dataset.pth')
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)

    # Parameter grouping (same as before)
    new_parameters = []
    for pname, p in model.named_parameters():
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
    optimizer = optim.Adam(params, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    total_epochs = 500
    batches_per_epoch = len(traindata_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    # Initialize RobustDiceLoss and SliceWiseMonitor
    loss_seg = RobustDiceLoss(alpha=0.5, beta=0.4, gamma=0.1)
    slice_monitor = SliceWiseMonitor(num_slices=32)

    writer = SummaryWriter('E:\\heart_seg\\Segmentation\\Seg_logs\\MNseg_logs9')
    train_time_sp = time.time()

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        log.info('Start epoch {}'.format(epoch))
        log.info('lr = {}'.format(scheduler.get_last_lr()))

        # Reset monitor at start of epoch
        slice_monitor.reset_epoch()

        for batch_id, batch_data in enumerate(traindata_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data
            volumes = volumes.to(device)

            optimizer.zero_grad()
            out_masks = model(volumes)
            out_masks = model(volumes)
            print("out_masks.shape:", out_masks.shape)

            # resize label
            [n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                label_mask = ndimage.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask
            new_label_masks = torch.tensor(new_label_masks).to(device)
            new_label_masks = new_label_masks.unsqueeze(1).float()

            # calculating loss
            loss_value_seg = loss_seg(out_masks, new_label_masks)
            dice = 1 - loss_value_seg
            loss = loss_value_seg

            loss.backward()
            optimizer.step()

            # Update slice monitor
            slice_monitor.update(out_masks, new_label_masks)

            epoch_loss += loss.item() * volumes.size(0)
            epoch_dice += dice.item() * volumes.size(0)
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))

        scheduler.step()

        # Log metrics
        epoch_loss = epoch_loss / len(traindata_loader.dataset)
        epoch_dice = epoch_dice / len(traindata_loader.dataset)
        writer.add_scalars('LOSS', {"train_loss": epoch_loss}, epoch + 1)
        writer.add_scalars('DICE', {"train_dice": epoch_dice}, epoch + 1)

        # Identify and log worst performing slices
        worst_slices = slice_monitor.get_worst_slices(k=3)
        for idx, (slice_name, avg_dice) in enumerate(worst_slices):
            writer.add_scalar(f'Worst_Slices/slice_{idx}', avg_dice, epoch + 1)
            print(f'Worst slice {slice_name}: Avg Dice = {avg_dice:.4f}')
        print('Epoch {}/{} Train_Loss: {:.4f} Train_Dice: {:.4f}'.format(
            epoch + 1, total_epochs, epoch_loss, epoch_dice))
        # Validation phase
        model.eval()
        val_slice_monitor = SliceWiseMonitor(num_slices=32)
        total_val_loss = 0
        total_val_dice = 0

        with torch.no_grad():
            for batch_id, val_batch_data in enumerate(valdata_loader):
                val_volumes, val_label_masks = val_batch_data
                val_volumes = val_volumes.to(device)
                val_out_masks = model(val_volumes)


                # resize label
                [n, _, d, h, w] = val_out_masks.shape
                new_val_masks = np.zeros([n, d, h, w])
                for label_id in range(n):
                    val_label_mask = val_label_masks[label_id]
                    [ori_c, ori_d, ori_h, ori_w] = val_label_mask.shape
                    val_label_mask = np.reshape(val_label_mask, [ori_d, ori_h, ori_w])
                    scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                    val_label_mask = ndimage.zoom(val_label_mask, scale, order=0)
                    new_val_masks[label_id] = val_label_mask

                new_val_masks = torch.tensor(new_val_masks).to(device)
                new_val_masks = new_val_masks.unsqueeze(1).float()

                # calculating loss
                val_loss = loss_seg(val_out_masks, new_val_masks)
                val_dice = 1 - val_loss

                # Update validation slice monitor
                val_slice_monitor.update(val_out_masks, new_val_masks)

                total_val_loss += val_loss.item() * val_volumes.size(0)
                total_val_dice += val_dice.item() * val_volumes.size(0)

            val_loss = total_val_loss / len(valdata_loader.dataset)
            val_dice = total_val_dice / len(valdata_loader.dataset)

            writer.add_scalars('LOSS', {"val_loss": val_loss}, epoch + 1)
            writer.add_scalars('DICE', {"val_dice": val_dice}, epoch + 1)

            # Log worst validation slices
            val_worst_slices = val_slice_monitor.get_worst_slices(k=3)
            for idx, (slice_name, avg_dice) in enumerate(val_worst_slices):
                writer.add_scalar(f'Val_Worst_Slices/slice_{idx}', avg_dice, epoch + 1)
                print(f'Val worst slice {slice_name}: Avg Dice = {avg_dice:.4f}')
            print('Epoch {}/{} Val_Loss: {:.4f} Val_Dice: {:.4f}'.format(
                epoch + 1, total_epochs, val_loss, val_dice))

        # Save model checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       'E:/heart_seg/Segmentation/Seg_models/model_epoch_{}.pth'.format(epoch + 1))
    print('Finished training')


if __name__ == '__main__':
    main()



