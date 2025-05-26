import os
import pandas as pd
#----------------------------------
# def rename_files_with_IM(folder_path, start_num):
#     # 获取文件夹中的所有文件
#     files = os.listdir(folder_path)
#     name_list=[]
#     name_list_new=[]
#     # 遍历文件夹中的所有文件
#     for index, file in enumerate(files):
#         # 检查文件名是否包含"IM"
#         if "IM" in file:
#             # 生成新的文件名
#             new_name = f"{start_num}{index+1}{os.path.splitext(file)[1]}"
#             name_list.append(os.path.basename(file))
#             name_list_new.append(new_name)
#             # 重命名文件
#             os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
#     name_record=list(zip(name_list,name_list_new))
#     rename_notes=pd.DataFrame(name_record)
#     rename_notes.to_excel('E:/Ultrasound_data/Classified_folder/negative/data_1.xlsx', index=False)
# # ---------------------------------------------------------------------------
# path = r"E:/Ultrasound_data/Classified_folder/negative/"
# files = os.listdir(path)
# rename_files_with_IM(path, "000")

# #------------------------------------------------------------------------
# def add_dcm_extension_to_files(input_folder):
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             old_file_path = os.path.join(root, file)
#             new_file_path = os.path.join(root, file + "P.dcm")
#             os.rename(old_file_path, new_file_path)
# #  #--------------------------------------------------------------------------
# # input_folder = "E:/Ultrasound_data/Classified_folder/negative/"
# # add_dcm_extension_to_files(input_folder)

folder_path = "D:\\Dicom_file\\MedicalNet\\original"  # 文件夹路径
prefix = ".dcm"

# 获取文件夹内所有文件的路径
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

# 遍历文件路径并重命名文件
for file_path in file_paths:
    file_name = os.path.basename(file_path)  # 获取文件名
    new_file_name =file_name + prefix
    new_file_path = os.path.join(folder_path, new_file_name)  # 新文件路径
    os.rename(file_path, new_file_path)
