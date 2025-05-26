import torch
import os
from torch import nn
from models import resnet_conseg
import torch.optim as optim
import nibabel as nib
import torch.nn.functional as F
from datasets.testdataset import MyDataset
from torch.utils.data import DataLoader
from utils.logger import log
import time
import numpy as np
from scipy import ndimage
import os
#----------------------------------------------------
def dice_loss(y_true, y_pred, epsilon=1e-6):
    intersection = 2.0 * torch.sum(y_true * y_pred) + epsilon
    union = torch.sum(y_true) + torch.sum(y_pred) + epsilon
    dice_coeff = intersection / union
    return 1.0 - dice_coeff

def convert_to_segmentation(output, threshold=0.5):
    # Apply sigmoid to convert output to probabilities
    probabilities = torch.sigmoid(output)

    # Apply threshold to get segmentation
    segmentation = torch.where(probabilities > threshold, torch.ones_like(probabilities), torch.zeros_like(probabilities))

    return segmentation
#-----------------------------------------------------
def main():
    val_list_path = 'D:\\Dicom_file\\MedicalNet\\preprocessed_image.txt'
    val_dir = "D:\\Dicom_file\\MedicalNet\\preprocessed"
    output_folder = "D:\\Dicom_file\\MedicalNet\\Seg_15model"
    val_dataset = MyDataset(root_dir=val_dir, img_list=val_list_path, input_D=16, input_H=224, input_W=224 ,
                                 phase='test')
    valdata_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,drop_last=False)

    model = resnet_conseg.resnet50(
        sample_input_W=224,
        sample_input_H=224,
        sample_input_D=16,
        shortcut_type='B',
        no_cuda=False,
        num_seg_classes=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0])
    print(model)
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(
        'D:\\pythonProject1\\Seg_models\\model15_epoch_400.pth'))
    pretrain = torch.load('D:\\pythonProject1\\Seg_models\\model15_epoch_400.pth',map_location=device)
    pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict)  # 字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict)  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    #------------------------------------------------------
    model.eval()  # switch model to evaluation mode
    with torch.no_grad():
        for batch_id, val_batch_data in enumerate(valdata_loader):
            val_volumes,file_name = val_batch_data
            val_volumes = val_volumes.cuda()
            val_out_masks = model(val_volumes)
            image = nib.load(file_name[0])
            matrix = image.affine
            original_name = os.path.basename(file_name[0])
            val_out_masks = torch.sigmoid(val_out_masks)
            #val_out_masks = convert_to_segmentation(val_out_masks)
            # Convert the tensor to numpy array
            val_out_masks_np = val_out_masks.cpu().squeeze().numpy()
            val_out_masks_np = val_out_masks_np.transpose((2,1,0))
            val_out_masks_np = np.rot90(val_out_masks_np, 2)
            flipped_val_out_masks_np = np.flip(val_out_masks_np, axis=0)
            flipped_val_out_masks_np = np.flip(flipped_val_out_masks_np, axis=1)
            val_out_masks_nii = nib.Nifti1Image(flipped_val_out_masks_np, matrix)
            #val_out_masks_nii = nib.Nifti1Image(val_out_masks_np, np.eye(4))
            # Create the output file name by adding the 'mask' suffix to the original image name
            output_file_name = os.path.join(output_folder,  original_name)

            # Save the image
            #nib.save(val_out_masks_nii, output_file_name)
if __name__ == '__main__':
    main()

