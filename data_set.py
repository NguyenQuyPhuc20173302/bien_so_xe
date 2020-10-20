import numpy as np
import cv2
from keras.utils import to_categorical
import os

IMG_SIZE = 224

data_name = {}


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def take_image(name_file):
    train_Image = []

    DIR = name_file

    name_label = []
    n = 0
    for name in os.listdir(DIR):
        name_label.append(name)
        folder = os.path.join(DIR, name)
        for img in os.listdir(folder):
            label = np.array([n])
            path = os.path.join(folder, img)
            image = read_image(path)
            train_Image.append([np.array(image), label])

        n += 1

    x = np.array([i[0] for i in train_Image]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array([i[1] for i in train_Image])

    y = to_categorical(y)
    return x, y


