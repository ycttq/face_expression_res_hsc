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
        im.append(struct.unpack_from('>784B', buf1, image_index)) # '>784B'的意思就是用大端法读取784个unsigned byte
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
    # print(im[1] ,im[2])
    label = get_label(label_data)

    bit = []
    with open("h_all_bit_test.csv", "w", newline='') as csvfile:###写入CSV,此处是将十进制按8位展开为bit
        writer = csv.writer(csvfile)
        for i in range(500,600):
            for a in im[i]:
                if a == 0:     ###a is tuple
                    for num in range(8):
                        bit.append(0)
                else:
                    str = '{:08b}'.format(a)
                    for k in range(8):
                        bit.append(int(str[k]))
            bit.insert(0,np.array(label[i])[0])
            print(len(bit),i)
            writer.writerow(bit)
            bit.clear()

