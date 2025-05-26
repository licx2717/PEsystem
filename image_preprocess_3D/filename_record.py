import os
import pandas as pd

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths



folder_path = "E:\\Ultrasound_data\\Classified_folder\\original"
file_paths = get_file_paths(folder_path)
image_names = [os.path.basename(path) for path in file_paths]
df = pd.DataFrame({"images_path": image_names})
df.to_excel('C:/Users/16921/Desktop/heart_records.xlsx', index=False)

