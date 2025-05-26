import os
import pydicom

def print_dicom_file_shapes(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            ds = pydicom.dcmread(filepath)
            print(f'{filename}: {ds.pixel_array.shape}')
        except Exception as e:
            print(f'Error loading {filename}: {e}')


# 使用示例
directory_path = 'D:\\dicom_test\\Seg_N'
print_dicom_file_shapes(directory_path)