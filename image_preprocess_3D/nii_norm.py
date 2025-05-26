# Import necessary libraries
import nibabel as nib
import os
import numpy as np

# Define function for normalization
def normalize(volume):
    """Normalize the volume"""
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    return volume

# Define the path for the input and output data
input_path = "F:\\Ultrasound_data\\Segmentation_folder\\Negative_label\\3059811"
output_path = "F:\\Ultrasound_data\\Segmentation_folder\\image_trans"

# Loop through all the files in the input folder
for file in os.listdir(input_path):
    if file.endswith(".nii.gz"):
        # Load the NIFTI image
        image = nib.load(os.path.join(input_path, file))
        #get affine and header(得到仿生矩阵和头文件)
        # 把仿射矩阵和头文件都存下来
        affine = image.affine.copy()
        hdr = image.header.copy()
        # Get the data from the image and normalize it
        data = image.get_fdata()
        data_new = normalize(data)
        # Create a new NIFTI image with the normalized data
        image_new = nib.Nifti1Image(data_new, affine, hdr)
        # Save the new image in the output folder with "_normalized" suffix in the name
        new_filename = f"{file}"
        nib.save(image_new, os.path.join(output_path, new_filename))
        print(new_filename)