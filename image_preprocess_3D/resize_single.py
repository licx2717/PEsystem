import os
import SimpleITK as sitk
# Define the new size and spacing


# def slice_nii_image(image, start_frame, end_frame):
#     array = sitk.GetArrayFromImage(image)
#     sliced_array = array[start_frame:end_frame]
#     sliced_image = sitk.GetImageFromArray(sliced_array)
#     return sliced_image


def nii_resize(file_path,output_path):
    image = sitk.ReadImage(file_path)
    resample = sitk.ResampleImageFilter()
    new_size = [800, 600, 32]
    resample.SetSize(new_size)
    original_spacing = image.GetSpacing()
    # Set the original spacing to the resample filter
    resample.SetOutputSpacing(original_spacing)
    resampled_image = resample.Execute(image)
    # Save the resampled image to the output directory
    filename = os.path.basename(file_path)
    output_file_name = os.path.join(output_path, filename)
    sitk.WriteImage(resampled_image, output_file_name)

file_path = "F:\\Ultrasound_data\\Segmentation_folder\\Negative_nii\\501256545_NIM_1415.nii.gz"
output_path = "F:\\Ultrasound_data\\Segmentation_folder\\Negative\\501256545"
nii_resize(file_path,output_path)

