import os
import random
import shutil


def select_and_copy_samples(positive_dir, negative_dir, output_positive_dir, output_negative_dir, num_samples=50):
    """
    从阳性和阴性样本中各随机挑选指定数量的样本，并复制到新的文件夹中。

    :param positive_dir: 原始阳性样本文件夹路径
    :param negative_dir: 原始阴性样本文件夹路径
    :param output_positive_dir: 保存新阳性样本的文件夹路径
    :param output_negative_dir: 保存新阴性样本的文件夹路径
    :param num_samples: 每种样本需要挑选的数量（默认为50）
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_positive_dir, exist_ok=True)
    os.makedirs(output_negative_dir, exist_ok=True)

    # 获取阳性样本和阴性样本的文件列表
    positive_samples = [f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))]
    negative_samples = [f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))]
    random.seed(42)

    selected_positive = random.sample(positive_samples, num_samples)
    selected_negative = random.sample(negative_samples, num_samples)

    # 将选中的样本复制到新文件夹
    for sample in selected_positive:
        src_path = os.path.join(positive_dir, sample)
        dst_path = os.path.join(output_positive_dir, sample)
        shutil.copy(src_path, dst_path)

    for sample in selected_negative:
        src_path = os.path.join(negative_dir, sample)
        dst_path = os.path.join(output_negative_dir, sample)
        shutil.copy(src_path, dst_path)

    print(f"已成功从阳性样本中挑选 {len(selected_positive)} 个样本到 {output_positive_dir}")
    print(f"已成功从阴性样本中挑选 {len(selected_negative)} 个样本到 {output_negative_dir}")


# 示例调用
positive_dir = "E:\\Ultrasound_data\\reselected\\test_dataset\\positive\\"  # 阳性样本文件夹路径
negative_dir = "E:\\Ultrasound_data\\reselected\\test_dataset\\negative\\"  # 阴性样本文件夹路径
output_positive_dir = "E:\\Ultrasound_data\\FGSM\\positive\\"  # 新阳性样本文件夹路径
output_negative_dir = "E:\\Ultrasound_data\\FGSM\\negative\\"  # 新阴性样本文件夹路径

select_and_copy_samples(positive_dir, negative_dir, output_positive_dir, output_negative_dir)

