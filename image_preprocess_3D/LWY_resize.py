import os
import numpy as np
import SimpleITK as sitk

#---------------------------------------------------------------------------
#数据格式转换
def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min )
    return volume
def reduce_dimensions(data):
    if np.ndim(data) == 5:
        data = data[:, :, :, :, 0]  # 保留前四个维度
    elif np.ndim(data) == 4:
        data = data[:, :, :, 0]  # 保留前三个维度
    return data
#----------------------------------------------
def data_transfrom(input_folder,output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".dcm"):
                images = os.path.join(root, file)
                dicom_image = sitk.ReadImage(images)
                start_index =[100, 50, 0]   # 起始索引
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
                new_size = [224, 224,16]
                resample = sitk.ResampleImageFilter()
                resample.SetOutputSpacing(new_spacing)
                resample.SetSize(new_size)
                resampled_image = resample.Execute(cropped_image)
                img_data = sitk.GetArrayFromImage(resampled_image)
                data_new = normalize(img_data)
                data_3d = reduce_dimensions(data_new)
                data_3d_image = sitk.GetImageFromArray(data_3d)
                filename=os.path.basename(file)
                #outfile = filename.split('_')[0]+".nii.gz"
                outfile = os.path.splitext(os.path.basename(filename))[0] + ".nii.gz"
                output_nifti_file = os.path.join(output_folder, outfile)
                sitk.WriteImage(data_3d_image, output_nifti_file)
                print(data_3d_image.GetSize())

input_path="E:\\LWYdata\\stack\\dicom\\"
output_path="E:\\LWYdata\\stack\\nii\\"

data_transfrom(input_path,output_path)