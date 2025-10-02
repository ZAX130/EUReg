import os
import shutil

root_directory = "E:/Documents/Dataset/ultrasound-datasets/CAMUS/camus/database/testing/"
new_root_directory = "D:/code/FVR-Net-main/data/datasets/CAMUS/testing/"
patient = 50
for i in range(1, patient + 1):
    if i < 10:
        patient_directory = root_directory + "patient000" + str(i) + "/"
    elif i < 100:
        patient_directory = root_directory + "patient00" + str(i) + "/"
    else:
        patient_directory = root_directory + "patient0" + str(i) + "/"

    for file_name in os.listdir(patient_directory):
        if "_sequence" in file_name:
            ori_file = os.path.join(patient_directory, file_name)
            new_file = os.path.join(new_root_directory, file_name)
            shutil.move(ori_file, new_file)