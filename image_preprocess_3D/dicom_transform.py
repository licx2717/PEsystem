import os
import glob
import pydicom
import nibabel as nib
import SimpleITK as sitk
import numpy as np
#-----------------------------------------
# def convert_dicom_to_npz(path):  # 将dicom转化为npz格式并存储在相同文件夹
#     # 遍历主文件夹
#     for subdir in os.listdir(path):
#         # 遍历其中子文件夹
#         subpath = os.path.join(path, subdir)
#         # 遍历子文件中所有的dicom格式文件
#         for dicom_file in glob.glob(os.path.join(subpath, "*.dcm")):
#             dicom_data = pydicom.dcmread(dicom_file)
#             # 格式转换
#             np_data = np.array(dicom_data.pixel_array)
#             # 将npz文件存储进输出文件夹
#             output_file = os.path.join(subpath, f"{subdir}.npz")
#             # Save the NumPy data to the output file
#             np.savez_compressed(output_file, data=np_data)
#-----------------------------------------------------------------------------------------
def convert_folder_to_nifti(input_dicom_folder, output_nifti_folder):
    for root, dirs, files in os.walk(input_dicom_folder):
        for file in files:
            if file.endswith(".dcm"):
                input_dicom_file = os.path.join(root, file)
                output_nifti_file = os.path.join(output_nifti_folder, file.replace(".dcm", ".nii.gz"))
                dicom_image = sitk.ReadImage(input_dicom_file)
                sitk.WriteImage(dicom_image, output_nifti_file)

input_dicom_folder="E:\\LWYdata\\stack\\dicom\\"
output_nifti_folder="E:\\LWYdata\\stack\\dcm2nii\\"
convert_folder_to_nifti(input_dicom_folder, output_nifti_folder)
# #------------------------------------------------------------
# def trans_3d(input_folder, output_folder):
#     for root,dirs,files in os.walk(input_folder):
#         for file in files:
#             if file.endswith("nii.gz"):
#                 input_folder=os.path.join(root,file)
#                 img = nib.load(input_folder)
#                 # Extract the 3D image data from the 5D image
#                 data_3d = img.get_fdata()[ :,:, :, 0, 0]
#                 # Create a new NIfTI image object with the 3D image data
#                 new_img = nib.Nifti1Image(data_3d, img.affine, img.header)
#                 # Save the new NIfTI image object to a new file
#                 output_file = os.path.join(output_folder, file.replace(".nii.gz", "_3d.nii.gz"))
#                 nib.save(new_img, output_file)
#
#
# input_folder = "E:/Ultrasound_data/Classified_folder/06_norm_nii_P/"
# output_folfer = "E:/Ultrasound_data/Classified_folder/07_3D_nii_p/"
# trans_3d(input_folder,output_folfer)
# #------------------------------------------------------------------------
# input_folder="D:\\nii_file"
# output_folfer="D:\\out"
# #trans_3d(input_folder,output_folfer)
#
# def save_first_three_dimensions(root_dir, new_dir):
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith("nii.gz"):
#                 filepath = os.path.join(dirpath, filename)
#                 img = nib.load(filepath)
#                 data = img.get_fdata()
#                 data_array = data[..., :3]  # keep only first three dimensions
#                 data_new = np.squeeze(data_array)[ :,:, :, 0]
#                 new_filepath = os.path.join(new_dir, filename)
#                 new_img = nib.Nifti1Image(data_new, img.affine, img.header)
#                 nib.save(new_img, new_filepath)
# #save_first_three_dimensions(input_folder,output_folfer)