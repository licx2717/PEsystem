import os
import SimpleITK as sitk

# Define the input and output directories
input_dir = "D:/input_dicom/"
output_dir = "D:/train/"

# Define the new size and spacing
new_size = [112, 112, 48]
new_spacing = [1.0, 1.0, 1.0]

# Loop through all files in the input directory
for file_name in os.listdir(input_dir):
    # Check if the file is a nii.gz file
    if file_name.endswith(".nii.gz"):
        # Load the image
        image = sitk.ReadImage(os.path.join(input_dir, file_name))

        # Create the resample filter
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)

        # Apply the filter
        resampled_image = resample.Execute(image)

        # Save the resampled image to the output directory
        output_file_name = os.path.join(output_dir, file_name)
        sitk.WriteImage(resampled_image, output_file_name)