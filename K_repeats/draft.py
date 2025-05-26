import os

def read_paths_and_labels_from_txt(file_path):
    images_path = []  # 存储图片路径
    images_label = []  # 存储图片标签
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                path = line.strip()  # 去除换行符和空白字符
                if os.path.exists(path):  # 检查路径是否存在
                    if "Positive" in path:
                        label = 1  # 如果路径包含 "Positive"，标签为 1
                    elif "Negative" in path:
                        label = 0  # 如果路径包含 "Negative"，标签为 0
                    else:
                        print(f"Unknown folder for path: {path}. Skipping.")
                        continue
                    images_path.append(path)
                    images_label.append(label)
                else:
                    print(f"The path {path} does not exist. Skipping.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return images_path, images_label
# 示例调用
txt_file_path = "D:\\MONAI-dev\\K_repeats\\val_records\\fold_1_train.txt"  # 替换为你的 .txt 文件路径
images_path, images_label = read_paths_and_labels_from_txt(txt_file_path)
# 打印结果
for path, label in zip(images_path, images_label):
    print(f"Path: {path}, Label: {label}")
