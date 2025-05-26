import SimpleITK as sitk
import os


def convert_dicom_to_nifti(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        if not dicom_names:
            #print(f"No DICOM files found in {folder_path}. Skipping.")
            continue

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # 提取特定级别的文件夹名称作为输出文件的名称
        parts = input_dir.split(os.sep)
        output_filename = parts + '.nii.gz'  # 假设特定级别的文件夹名称是倒数第三个元素
        output_file_path = os.path.join(output_dir, output_filename)
        # 检查输出目录中是否已存在相同文件名的文件
        count = 1
        while True:
            if count == 1:
                candidate_filename = output_filename
            else:
                candidate_filename = f"{parts[-3]}_{count}.nii.gz"
            output_file_path = os.path.join(output_dir, candidate_filename)
            if not os.path.exists(output_file_path):
                break
            count += 1

        sitk.WriteImage(image, output_file_path)
        #print(f"Converted {folder} to NIfTI format.")

def process_all_samples(root_dir, output_base_dir):
    for sample_id in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue
        for subdir, dirs, files in os.walk(sample_path):
            for dir in dirs:
                dicom_dir = os.path.join(subdir, dir)
                print(f"Processing: {dicom_dir}")
                convert_dicom_to_nifti(dicom_dir, output_base_dir)
    print("完成所有转换。")

folder_path = 'E:\\LWYdata\\stack\\dicom\\'
output_dir = 'E:\\LWYdata\\stack\\nii\\'

process_all_samples(folder_path, output_dir)
