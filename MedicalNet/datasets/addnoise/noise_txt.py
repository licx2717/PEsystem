import os
import glob

image_dir = "E:\\Ultrasound_data\\Dicom_file\\MedicalNet\\Stability_dcm_mask\\Stability_dcm\\"  # Your directory path
output_file = "E:\\Ultrasound_data\\Dicom_file\\MedicalNet\\Stability_dcm_mask.txt"  # Output text file

# Get all .nii.gz files recursively from subdirectories
nii_paths = glob.glob(os.path.join(image_dir, '**', '*.nii.gz'), recursive=True)

# Write paths to the text file
with open(output_file, 'w') as f:
    for path in nii_paths:
        # Write the absolute path for each file
        f.write("%s\n" % os.path.abspath(path))

print(f"Found {len(nii_paths)} .nii.gz files. Paths written to {output_file}")
