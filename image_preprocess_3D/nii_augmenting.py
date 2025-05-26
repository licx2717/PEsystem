import nibabel as nib
import random
from scipy import ndimage
import os

input_dir = 'D:/dicom/'
output_dir = 'D:/out/'
# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.nii.gz'):
        # Load the image data
        img = nib.load(os.path.join(input_dir, filename))
        data = img.get_fdata()
        angle=random.randint(-30,30)
        img_rote = ndimage.rotate(data, angle)
        new_filename = filename.replace('.nii.gz', '_rotated.nii.gz')
        nib.save(nib.Nifti1Image(img_rote, img.affine), os.path.join(output_dir, new_filename))