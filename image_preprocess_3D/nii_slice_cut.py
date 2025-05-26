import nibabel as nib
import os

# Set the directory path to where the nii images are stored
dir_path ='E:/Ultrasound_data/Classified_folder/negative_nii/'

# Loop through the directory and apply the cropping to each nii image
for filename in os.listdir(dir_path):
    if filename.endswith(".nii.gz"):
        # Load the image and get the image data
        img = nib.load(os.path.join(dir_path, filename))
        img_data = img.get_fdata()

        # Perform the cropping operation on the image data
        cropped_data = img_data[100:700, 50:550, 0:48]

        # Create a new nibabel image with the cropped data and save it to disk
        cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        new_filename = f"cropped_{filename}"
        nib.save(cropped_img, os.path.join(dir_path, new_filename))