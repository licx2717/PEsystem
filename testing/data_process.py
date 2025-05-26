import os
import random
import json
import shutil
import pickle
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------
def copy_random_files(folder_path, destination_folder, num_files=1000, seed=123):
    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)
    # 设置随机种子
    random.seed(seed)
    # 随机选择指定数量的文件
    random_files = random.sample(file_list, num_files)
    # 将选中的文件复制到另一个文件夹
    for file_name in random_files:
        source = os.path.join(folder_path, file_name)
        destination = os.path.join(destination_folder, file_name)
        shutil.copyfile(source, destination)

#---------------------------------------------------------------------------
#数据格式转换
def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    return volume

def data_transfrom(input_folder,output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".dcm"):
                images = os.path.join(root, file)
                dicom_image = sitk.ReadImage(images)
                start_index = [100, 50, 0]  # 起始索引
                size = [600, 500, 48]  #
                filter = sitk.RegionOfInterestImageFilter()
                filter.SetIndex(start_index)
                filter.SetSize(size)
                try:
                    cropped_image = filter.Execute(dicom_image)
                except RuntimeError:
                    cropped_image = dicom_image
                    print(f"Filename{file} .The file size is not suitable for cropping.")
                new_spacing = [1, 1, 1]
                new_size = [224, 224, 16]
                resample = sitk.ResampleImageFilter()
                resample.SetOutputSpacing(new_spacing)
                resample.SetSize(new_size)
                resampled_image = resample.Execute(cropped_image)
                img_data = sitk.GetArrayFromImage(resampled_image)
                data_new = normalize(img_data)
                data_3d = data_new[:, :, :, 0]
                data_3d_image = sitk.GetImageFromArray(data_3d)
                output_nifti_file = os.path.join(output_folder, file.replace(".dcm", ".nii.gz"))
                sitk.WriteImage(data_3d_image, output_nifti_file)
                print(data_3d_image.GetSize())
#---------------------------------------------------------------------------

def read_data(root: str, val_rate: float = 0):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".gz"]  # 支持的文件后缀类型[[image, label], []]
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} were found in the dataset.".format(len(train_images_path)))

    return train_images_path, train_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = 'class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list




