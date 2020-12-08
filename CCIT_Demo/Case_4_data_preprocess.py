import os
import numpy as np
import tarfile
import csv
from PIL import Image

## init variable
#dataset_directory = "F:\\kaggle\\microsoft_malware"
dataset_directory = "G:\\kaggle_temp"
train_path = "{0}\\{1}".format(dataset_directory, "train")
test_path = "{0}\\{1}".format(dataset_directory, "test")
train_csv_path = "{0}\\{1}".format(dataset_directory, "trainLabels.csv")

resize_image = 1024

def _Padding(data, width = resize_image):
    padding_length = width - (len(data) % width)
    
    if padding_length != width:
        data = np.pad(data, (0, padding_length), 'constant')

    return data

def _ReadByte(f, count):
    return f.read(count)

def _Token(file_path):
    file_data = []
    cnt = 0

    with open(file_path, 'rb') as f:
        while True:
            if len(_ReadByte(f, 8)) == 0:
                break

            stop = False

            for i in range(0, 16):
                eat_string = _ReadByte(f, 1)

                if eat_string != b" ":
                    stop = True
                    break

                check = _ReadByte(f, 2)

                try:
                    file_data.append(int(check, 16))
                except:
                    file_data.append(0x00)

            if stop == True:
                break

            lines = _ReadByte(f, 2)
            cnt += 1

    return file_data

def _BuildBinaryImage(file_directory):
    check = False

    for (dir_path, dir_names, file_names) in os.walk(file_directory):
        for index, value in enumerate(file_names):
            if check == True:
                file_path = "{0}\\{1}".format(file_directory, value)
                image_path = "{0}_image\\{1}.jpg".format(file_directory, value)

                file_data = _Token(file_path)
                file_data = np.asarray(file_data, dtype='uint8')
                file_data = _Padding(file_data)
                file_data = np.reshape(file_data, (-1, resize_image))
                file_image = Image.fromarray(file_data, 'L')
                file_image = file_image.resize((resize_image, resize_image))
                file_image.save(image_path)
                print ("[*] {0} OK.".format(file_path))

def _BuildLabelDirectory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def BuildBinaryImage():
    #_BuildBinaryImage(train_path)
    _BuildBinaryImage(test_path)

def BuildLabel():
    ## Build label directory
    for i in range(1, 10):
        _BuildLabelDirectory("{0}_image\\{1}\\".format(train_path, str(i)))

    ## Move training image
    with open(train_csv_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            org_path = "{0}_image\\{1}.bytes.jpg".format(train_path, row['Id'])
            dst_path = "{0}_image\\{1}\\{2}.bytes.jpg".format(train_path, row['Class'], row['Id'])
            os.rename(org_path, dst_path)