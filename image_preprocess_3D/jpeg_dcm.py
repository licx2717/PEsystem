import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
from PIL import Image
import datetime
import os


def jpeg_to_dicom(jpeg_files, output_dicom):
    # 创建DICOM数据集
    ds = FileDataset(output_dicom, {}, file_meta=Dataset(), preamble=b"\0" * 128)

    # 设置一些基本的DICOM元数据
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "OT"  # 设置为其他模式
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')

    # 读取JPEG图像并将它们作为DICOM像素数据
    images = []
    for jpeg_file in jpeg_files:
        img = Image.open(jpeg_file)
        img = img.convert("L")  # 转换为灰度图
        images.append(np.array(img))

    # 将图像数据合并为3D数组 (frames, height, width)
    pixel_array = np.stack(images)

    # 设置DICOM的像素数据
    ds.PixelData = pixel_array.tobytes()
    ds.Rows, ds.Columns = pixel_array.shape[1], pixel_array.shape[2]
    ds.NumberOfFrames = pixel_array.shape[0]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # 保存为DICOM文件
    ds.save_as(output_dicom)
    print(f"保存DICOM文件至 {output_dicom}")


# 使用示例
jpeg_files = [f"E:\\LWYdata\\stack\\resized_jpeg\\frame_{i}.jpg" for i in range(16)]  # 假设你的16帧JPEG文件命名为 frame_0.jpeg 到 frame_15.jpeg
output_dicom = "E:\\LWYdata\\stack\\dicom\\resized_3.dcm"
jpeg_to_dicom(jpeg_files, output_dicom)
