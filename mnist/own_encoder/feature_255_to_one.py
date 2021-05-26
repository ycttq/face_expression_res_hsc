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
        # im.append(np.reshape(temp,(28,28)))
        im.append(temp)
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
    print(np.array(label[1])[0])
    # for i in range(50):#### 展示图数据
    #     plt.subplot(10 , 5, i + 1)
    #     # title =str(label[i])+'origine --image'
    #     print(i)
    #     # plt.title(title, fontproperties='SimHei')
    #     plt.imshow(im[i], cmap='gray')
    # plt.show()


    # e_h = []   ###获取Hilbert还原图
    # d_h = []
    # for i in range(50):
    #     e_h.append(encode(im[i], 28, 2))
    #     d_h.append(decode(e_h[i], 28, 2))
    #     plt.subplot(10, 5, i + 1)
    #     # title =str(label[i])+ "hilbert transform --image"
    #     # plt.title(title, fontproperties='SimHei')
    #     plt.imshow(d_h[i], cmap='gray')
    # plt.show()

    # file = open("h_train---image.idx-ubyte","wb")
    # count = 1  ####写入二进制文件
    # temp = struct.pack('4I',count,count,count,count)
    # for i in range(60000):
    #     e_h = encode(im[i], 28, 1)
    #     for w in range(28):
    #         temp = struct.pack('I',e_h[w])
    #         file.write(temp)
    #     count = i
    #     print(count)
    # file.close()


    with open("h_bit_all_train.csv", "w",newline ='') as csvfile:###写入CSV,此处是hilbert 本质是另gray图中 255 直接转化为1，其他为0
        writer = csv.writer(csvfile)
        for i in range(50000): ####划分开关
            bit = []
            for a in im[i]:
                if a == 255:  ###a is tuple
                    bit.append(1)
                else:
                    bit.append(0)
            bit.insert(0,np.array(label[i])[0])
            print(i)
            print(len(bit))
            writer.writerow(bit)
            bit.clear()






