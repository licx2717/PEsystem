import os
import SimpleITK as sitk
from PIL import Image

def convert_dicom_to_jpeg(input_folder, output_folder, frame_numbers):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为DICOM格式
        if filename.endswith('.dcm'):
            # 读取DICOM文件
            ds = sitk.ReadImage(os.path.join(input_folder, filename))

            # 遍历指定的帧数
            for frame_number in frame_numbers:
                # 检查帧数是否在范围内
                if frame_number >= 0 and frame_number < ds.GetDepth():
                    # 获取指定帧的像素数据
                    image = sitk.GetArrayViewFromImage(ds)[frame_number]

                    # 创建图像对象
                    image = Image.fromarray(image)

                    # 构建输出文件路径
                    output_filename = f"{filename}_{frame_number}.jpg"
                    output_path = os.path.join(output_folder, output_filename)

                    # 保存图像为JPEG格式
                    image.save(output_path)

input_folder = "E:\\胸腔积液阴性"
output_folder = "D:\\JPEG"
frame_numbers = [5, 15]

convert_dicom_to_jpeg(input_folder, output_folder, frame_numbers)
