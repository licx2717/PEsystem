import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet
from utils_record import read_split_data, plot_data_loader_image

# http://download.tensorflow.org/example_images/flower_photos.tgz
#root = "D:/ultrasound_re/dataset/"  # 数据集所在根目录
root=""
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root,Rotate=False)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataload workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images, labels = data


if __name__ == '__main__':
    main()
    print('Success')
