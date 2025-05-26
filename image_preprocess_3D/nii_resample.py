import os
import SimpleITK as sitk

# Set the input and output directories
input_dir ="E:/Ultrasound_data/Classified_folder/03_Cropped_nii_P/"
output_dir ="E:/Ultrasound_data/Classified_folder/04_Resample_nii_P/"

# Set the new spacing
new_spacing = [1, 1, 1] # set the new spacing here

# Loop through all the files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".nii.gz"):
        # Load the image
        image = sitk.ReadImage(os.path.join(input_dir, file_name))

        # Resample the image to a new spacing
        resampled_image = sitk.Resample(image, new_spacing)

        # Normalize the image
        normalized_image = sitk.Normalize(resampled_image)

        # Save the normalized image
        output_file_name = os.path.join(output_dir, file_name)
        sitk.WriteImage(normalized_image, output_file_name)