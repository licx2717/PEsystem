import pydicom
import numpy as np

def load_3d_dicom(file_path):
    """加载包含 3D 数据的 DICOM 文件"""
    ds = pydicom.dcmread(file_path)
    if hasattr(ds, 'PixelData'):
        volume = ds.pixel_array  # 提取体数据
        return volume, ds
    else:
        raise ValueError("DICOM文件不包含像素数据（Pixel Data）。")

def global_3d_hist_equalization(volume):
    """对整个3D体积进行直方图均衡化"""
    hist, bin_edges = np.histogram(volume.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # 归一化
    lut = np.interp(np.arange(0, 256), bin_edges[:-1], cdf_normalized * 255).astype(np.uint16)
    equalized_volume = lut[volume]
    return equalized_volume.astype(np.uint16)
def slicewise_2d_hist_equalization(volume):
    """逐层2D直方图均衡化"""
    equalized_volume = np.zeros_like(volume)
    for z in range(volume.shape[0]):
        slice_img = volume[z]
        hist = np.histogram(slice_img, bins=256, range=(0, 256))[0]
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        lut = (cdf_normalized * 255).astype(np.uint16)
        equalized_volume[z] = lut[slice_img]
    return equalized_volume

from skimage import exposure

def clahe_3d(volume, kernel_size=(64, 64, 64), clip_limit=0.03):
    """3D CLAHE"""
    return exposure.equalize_adapthist(
        volume.astype(np.float32) / np.max(volume),  # 归一化到[0,1]
        kernel_size=kernel_size,
        clip_limit=clip_limit
    ) * np.max(volume)  # 恢复原始数值范围




import matplotlib.pyplot as plt

# 1. 加载 DICOM 文件
dicom_file = "D:\\pythonProject1\\dataset\\denosie\\before\\N_003.dcm"
volume_3d, ds = load_3d_dicom(dicom_file)

# 2. 预处理：确保数值范围正确
volume_3d = volume_3d.astype(np.float32)
volume_3d = np.clip(volume_3d - np.min(volume_3d), 0, None)

# 3. 全局直方图均衡化
equalized_global = global_3d_hist_equalization(volume_3d.astype(np.uint16))

# 4. 显示结果
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
z_slice = volume_3d.shape[0] // 2  # 取中间切片

# 原始图像和直方图
axs[0, 0].imshow(volume_3d[z_slice], cmap='gray')
axs[0, 0].set_title('Original Slice')
axs[0, 1].hist(volume_3d.flatten(), bins=256, range=(0, 256))
axs[0, 1].set_title('Original Histogram')

# 全局均衡化结果
axs[1, 0].imshow(equalized_global[z_slice], cmap='gray')
axs[1, 0].set_title('Global Equalized Slice')
axs[1, 1].hist(equalized_global.flatten(), bins=256, range=(0, 256))
axs[1, 1].set_title('Global Equalized Histogram')

plt.tight_layout()
plt.show()

