
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import nibabel as nib



class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)


    # def __nii2tensorarray__(self, data):
    #     [x, y, z] = data.shape
    #     new_data = np.reshape(data, [1, x, y, z])
    #     new_data = new_data.transpose((0, 3, 2, 1))
    #     new_data = new_data.astype("float32")
    #     return new_data

    def __nii2tensorarray__(self, data):
        [x, y, z] = data.shape
        new_data = np.reshape(data, [z, y, x])  # 直接使用[z, y, x]，省略通道维度
        new_data = new_data.astype("float32")
        return new_data

    def __getitem__(self, item):
        image_array = nib.load(self.images_path[item]).get_fdata()
        #image = image_array[:, :, :,0,0]
        image = image_array[:, :, :]
        img = self.__nii2tensorarray__(image)

        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

