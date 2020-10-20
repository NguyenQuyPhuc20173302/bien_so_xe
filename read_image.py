import cv2
import os
import matplotlib.pyplot as plt


# hàm để hiển thị kết quả dự đoán lên colab trực tiếp
def x2(image):
    original_width, original_height = image.shape[1], image.shape[0]
    resized_image = cv2.resize(image, (2 * original_width, 2 * original_height), interpolation=cv2.INTER_CUBIC)

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 10))
    plt.axis("off")
    plt.imshow(resized_image)
    plt.show()


def save(path, name):
    anh = cv2.imread(path)
    gray = cv2.cvtColor(anh, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imwrite(name, thresh)


path_1 = 'charTrainset'
path_2 = 'data_set'

for name in os.listdir(path_1):
    '''luu anh'''
    save_ = os.path.join(path_2, name)

    ten_nhan = os.path.join(path_1, name)

    for image in os.listdir(ten_nhan):
        anh = os.path.join(ten_nhan, image)
        name_save = os.path.join(save_, image)
        save(anh, name_save)
