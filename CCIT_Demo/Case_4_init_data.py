import os
import numpy as np
import tarfile
import csv
import pandas as pd
from PIL import Image

## init variable
dataset_directory = "G:\\kaggle_temp"
train_path = "{0}\\{1}".format(dataset_directory, "train.csv")
train_image = "{0}\\{1}".format(dataset_directory, "train_image")
test_path = "{0}\\{1}".format(dataset_directory, "test.csv")
test_image = "{0}\\{1}".format(dataset_directory, "test_image")
sample_path = "{0}\\{1}".format(dataset_directory, "sampleSubmission.csv")
x_train_pickle_path = "{0}\\{1}".format(dataset_directory, "x_train_data.npy")
y_train_pickle_path = "{0}\\{1}".format(dataset_directory, "y_train_data.npy")
x_test_pickle_path = "{0}\\{1}".format(dataset_directory, "x_test_data.npy")

img_width, img_height = 128, 128

cols_type = {}

cols_read = []

def _build_train_test_csv(build_csv = True):
    cols_type.update({'Id':'category', 'Class':'uint8'})

    for y in range(0, img_height):
        for x in range(0, img_width):
            column_name = "{0}_{1}".format(y, x)
            cols_type.update({column_name:'uint8'})
            cols_read.append(column_name)

    if build_csv == True:
        with open(test_path, "w") as f:
            cnt = 0

            # header
            f.write('Id')
            for idx, val in enumerate(cols_read):
                f.write(",{0}".format(val))
            f.write('\n')

            with open(sample_path, "r") as csvfile:
                rows = csv.reader(csvfile)
                next(rows, None)

                for row in rows:
                    file_name = "{0}.bytes.jpg".format(row[0])
                    img = Image.open("{0}\\{1}".format(test_image, file_name))
                    img = img.resize((img_width, img_height))
                    img_data = np.asarray(img, dtype='uint8')

                    f.write(row[0])
                    for y in range(0, img_height):
                        for x in range(0, img_width):
                            f.write(",{0}".format(img_data[y, x]))

                    f.write('\n')
                    print("[*] test file. ({0})".format(cnt))
                    cnt += 1

        with open(train_path, "w") as f:
            cnt = 0

            # header
            f.write('Id,Class')
            for idx, val in enumerate(cols_read):
                f.write(",{0}".format(val))
            f.write('\n')

            # raw_data
            for i in range(1, 10):
                for file_name in os.listdir("{0}\\{1}".format(train_image, i)):
                    file_id = file_name.replace(".bytes.jpg", "")
                    f.write("{0},{1}".format(file_id, i))
                    img = Image.open("{0}\\{1}\\{2}".format(train_image, i, file_name))
                    img = img.resize((img_width, img_height))
                    img_data = np.asarray(img, dtype='uint8')

                    for y in range(0, img_height):
                        for x in range(0, img_width):
                            f.write(",{0}".format(img_data[y, x]))

                    f.write('\n')
                    print("[*] train file. ({0})".format(cnt))
                    cnt += 1

    print("[+] Build train and test data (.csv) successfully.\n")

def load_bin_data():
    print ("[*] Read train_x data to {0}".format(x_train_pickle_path))
    train_x = np.load(x_train_pickle_path)
    print ("[*] Read train_y data to {0}".format(y_train_pickle_path))
    train_y = np.load(y_train_pickle_path)
    print ("[*] Read test_x data to {0}".format(x_test_pickle_path))
    test_x = np.load(x_test_pickle_path)

    print("[+] Load data sucessfully.\n")
    return train_x, train_y, test_x

def load_data(save_pickle=True, build_csv = False):
    _build_train_test_csv(build_csv)

    print ("[*] Read csv data from {0} for training data".format(train_path))
    train_x = pd.read_csv(train_path, dtype=cols_type, usecols=cols_read)

    print ("[*] Read csv data from {0} for training label".format(train_path))
    train_y = pd.read_csv(train_path, dtype=cols_type, usecols=['Class'])

    print ("[*] Read csv data from {0} from testing data".format(test_path))
    test_x = pd.read_csv(test_path, dtype=cols_type, usecols=cols_read)

    if save_pickle == True:
        print ("[*] Write train_x data to {0}".format(x_train_pickle_path))
        np.save(x_train_pickle_path, train_x)
        print ("[*] Write train_y data to {0}".format(y_train_pickle_path))
        np.save(y_train_pickle_path, train_y)
        print ("[*] Write test_x data to {0}".format(x_test_pickle_path))
        np.save(x_test_pickle_path, test_x)

    return train_x.values, train_y.values, test_x.values