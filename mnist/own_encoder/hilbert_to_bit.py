#coding==utf_8
import numpy as np
import struct
import matplotlib.pyplot as plt
from hilbert import encode,decode
import pandas as pd
import csv

def readfile():
    with open('train-images.idx3-ubyte', 'rb') as f1:
        buf1 = f1.read()
    with open('train-labels.idx1-ubyte', 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1):
    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(60000):  ##读取图片数量
        temp = struct.unpack_from('>784B', buf1, image_index) # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(np.reshape(temp,(28,28)))
        image_index += struct.calcsize('>784B')  # 每次增加784B
    return im


def get_label(buf2): # 得到标签数据
    label = []
    label_index = 0
    label_index += struct.calcsize('>II')
    for i in range(60000):
        # temp = struct.unpack_from('>B', buf2, label_index)
        label.append(struct.unpack_from('>B', buf2, label_index))
        label_index += struct.calcsize('>B')

    return label


if __name__ == "__main__":
    image_data, label_data = readfile()


    im = get_image(image_data)

    label = get_label(label_data)
    plt.title("original image")
    plt.imshow(im[0],cmap='gray')
    plt.axis('off')
    plt.show()
    print(np.array(label[1])[0])


    with open("h_bit_all_train.csv", "w",newline ='') as csvfile:###写入CSV,此处是hilbert  将不是0的部分全部化为1，呈现出基本的形状，
        writer = csv.writer(csvfile)
        for i in range(1): ####划分开关
            e_d = encode(im[i],28,1)
            print(e_d)
            plt.imshow(e_d,cmap="gray")
            plt.show()
            # d_h = decode(e_d,28,1)
            # _z = np.insert(np.array(d_h).flatten(), 0, label[i])
            # writer.writerow(_z)