import monai
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, RandFlip,AddChannel,
    ScaleIntensityd, Spacingd, RandCropByPosNegLabeld, RandAffined, RandRotate)
from torchvision import transforms
import nibabel as nib
import SimpleITK as sitk
import os
import monai
from dataload.utils import read_split_data
from dataload.my_dataset import MyDataSet

data_path="D:\\train_data"
Train_images_path, Train_images_label, val_images_path, val_images_label = read_split_data(data_path, val_rate=0.5)


transform_1= monai.transforms.Compose([transforms.ToTensor(),
                                       AddChannel()])
transform_2= monai.transforms.Compose([AddChannel()])
transform_3=monai.transforms.Compose([transforms.ToTensor()])
train_data_set_1 = MyDataSet(images_path=Train_images_path,
                           images_class=Train_images_label,
                           transform=transform_1)
for image, label in train_data_set_1:
    print(image.shape)
train_data_set_2 = MyDataSet(images_path=Train_images_path,
                           images_class=Train_images_label,
                           transform=transform_2)
for image, label in train_data_set_2:
    print(image.shape)
train_data_set_3 = MyDataSet(images_path=Train_images_path,
                           images_class=Train_images_label,
                           transform=transform_3)
for image, label in train_data_set_3:
    print(image.shape)


