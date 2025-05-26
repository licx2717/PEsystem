import os

# 读取txt文件并打印内容
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(content)
            return content.splitlines()
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def modify_folder_name(paths, old_folder_name, new_folder_name):
    modified_paths = []
    for path in paths:
        if old_folder_name in path:
            modified_path = path.replace(old_folder_name, new_folder_name)
            modified_paths.append(modified_path)
        else:
            modified_paths.append(path)
    return modified_paths

def save_modified_paths(file_path, modified_paths):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for path in modified_paths:
                file.write(path + '\n')
        print(f"Modified paths have been saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


filepath = "D:\\MONAI-dev\\K_repeats\\folds_records\\fold_1_val_paths.txt"
old_folder_name = r"E:\HeartWound\project_pretest\data\train_data"
new_folder_name = r"E:\Ultrasound_data\Val_records"
output_filepath="D:\\MONAI-dev\\K_repeats\\val_records\\fold_1_val.txt"
# 读取文件内容
paths = read_txt_file(filepath)

# 修改文件夹名称
if paths:
    modified_paths = modify_folder_name(paths, old_folder_name, new_folder_name)
    for path in modified_paths:
        print(path)
    save_modified_paths(output_filepath, modified_paths)