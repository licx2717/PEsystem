# First, we need to import the necessary libraries
import os
import nibabel as nib

# Next, we need to define the path to the directory containing the NII files
directory_path = "D:\\dicom_test\\class_N"

# Then, we can use a for loop to iterate through all NII files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".nii.gz"):
        # Load the image using nibabel
        img = nib.load(os.path.join(directory_path, filename))
        # Get the shape of the image
        img_shape = img.shape
        # Print the shape of the image
        print(f"The shape of {filename} is {img_shape}")