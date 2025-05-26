import nibabel as nib
from scipy.ndimage import zoom


def resize_nii_image(file_path, new_shape=(112, 112, 48)):
    # Load the image
    img = nib.load(file_path)
    data = img.get_fdata()

    # Calculate the resize factor
    resize_factor = [n / o for n, o in zip(new_shape, data.shape)]

    # Resize the image
    resized_data = zoom(data, resize_factor)

    # Create a new Nifti1Image with the resized data and the original image's affine
    resized_img = nib.Nifti1Image(resized_data, img.affine)

    return resized_img

resized_img = resize_nii_image("D:/input_dicom/AGBC_001_0000.nii.gz")
nib.save(resized_img, "D:/train/AGBC_001_resize.nii.gz")