import pydicom


def read_dicom_info(dicom_file_path):
    """
    读取DICOM文件中的基本信息。

    参数:
    - dicom_file_path: DICOM文件的路径

    返回:
    - 包含DICOM元数据信息的字典
    """
    ds = pydicom.dcmread(dicom_file_path)

    info = {
        'PatientName': ds.PatientName,
        'PatientID': ds.PatientID,
        'Modality': ds.Modality,
        'StudyInstanceUID': ds.StudyInstanceUID,
        'SeriesInstanceUID': ds.SeriesInstanceUID,
        'SOPInstanceUID': ds.SOPInstanceUID,
        'SOPClassUID': ds.SOPClassUID,
        'StudyDate': ds.StudyDate,
        'StudyTime': ds.StudyTime,
        'Rows': ds.Rows,
        'Columns': ds.Columns,
        'NumberOfFrames': ds.NumberOfFrames,
        'BitsAllocated': ds.BitsAllocated,
        'BitsStored': ds.BitsStored,
        'HighBit': ds.HighBit,
        'SamplesPerPixel': ds.SamplesPerPixel,
        'PhotometricInterpretation': ds.PhotometricInterpretation
    }

    return info


# 示例用法
input_dicom = "D:\\Data\\heart_raw\\N_001.dcm"  # 这里替换为实际的DICOM文件路径
info = read_dicom_info(input_dicom)
print(info)
