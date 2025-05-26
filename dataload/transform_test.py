import monai.transforms as transform
import os
import nibabel as nib
from monai.transforms import Compose,RandFlip, RandAffine,RandRotate,SaveImage

input_path="D:\\input_dicom"
output_path="D:\\train"

transforms = Compose([
    #RandAffine(prob=1.0),
    RandRotate(prob=1.0),
    #RandAffine(prob=0.5, translate_range=(4, 10, 10), padding_mode="border"),
   RandFlip(prob=0.5),

    #RandAffine(prob=1.0,spatial_size=(256, 256),translate_range=(40, 40, 2),scale_range=(0.15, 0.15, 0.15),padding_mode="border"),
    SaveImage(output_dir=output_path)
])
nii_files = [file for file in os.listdir(input_path) if file.endswith('.nii.gz')]
for nii_file in nii_files:
    nii_path = os.path.join(input_path, nii_file)
    nii_data = nib.load(nii_path).get_fdata()
    nii_data=transform.AddChannel()(nii_data)
    transforms(nii_data)