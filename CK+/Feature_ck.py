#coding = utf_8
from collections import Counter
import os
import dlib
import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt
from hilbert import encode,decode
def more_static_num(data_array,label): ####多分布的直方图
    bit = []
    for j in range(0, 48, 8):
        for k in range(0, 48, 8):
            b = data_array[j:j + 8, k:k + 8]  # [行开始:行结束,列开始:列结束]
            b_one = np.ndarray.flatten(b).tolist()
            # print(b_one)
            b_one.sort()
            cl = Counter(b_one)
            for value, times in cl.items():  #### value是值 times是重复的次数
                for i in range(times):
                    str = '{:08b}'.format(value)
                    for a in range(8):
                        bit.append(int(str[a]))
    bit.insert(0,label)
    return bit
def static_num(data_list,label):  ####lbp统计直方图
    bit = []
    data_list.sort()
    cl = Counter(data_list)
    for value, times in cl.items():  #### value是值 times是重复的次数
        for i in range(times):
            str = '{:08b}'.format(value)
            for a in range(8):
                bit.append(int(str[a]))
    bit.insert(0, label)
    return bit
def ten_to_bit(data_list,label):
    bit = []
    for i in data_list:
        if i == 0:
            for num in range(8):
                bit.append(0)
        else:
            str = '{:08b}'.format(i)
            for k in range(8):
                bit.append(int(str[k]))
    bit.insert(0, label)
    return bit
def hilbert_bit(gray_matrix,label):
    bit = []
    e_h = encode(gray_matrix,48,1)
    d_h=  decode(e_h,48,1)
    bit[0:] = np.ndarray.flatten(d_h)
    bit.insert(0,label)
    return bit
def Non_zero_to_0ne(data_list,label):
    bit = []
    for i in data_list:
        if i == 0:
            bit.append(0)
        else:
            bit.append(1)
    bit.insert(0, label)
    return bit
def write_to_file(data,labels):
    rows = 0  ####写入训练集和测试集
    with open("face_all_bit_test.csv", "w", newline='') as test_file, open("face_all_bit_train.csv", "w",
                                                                           newline='') as train_file:
        writer_test = csv.writer(test_file)
        writer_train = csv.writer(train_file)
        for count in range(len(data)):
            rows = rows + 1
            if rows % 5 == 0:
                writer_test.writerow(more_static_num(data[count], labels[count]))
            else:
                writer_train.writerow(Non_zero_to_0ne(data[count], labels[count]))

def image_pre():
    # 模型路径 以及文件路径
    data_path = ['ori_data/anger','ori_data/contempt','ori_data/disgust','ori_data/fear','ori_data/happy','ori_data/sadness','ori_data/surprise']
    emotion = ['anger','contempt','disgust','fear','happy','sadness','surprise']
    emotion_bit = ['0','1','2','3','4','5','6']
    output = 'feature_bit_data'
    predictor_path = 'shape-predictor/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    data = []
    label = []
    # for root,dir,files in os.walk(data_path,topdown=True):
    #     print(root)
    for files_count in range(len(data_path)):
        files = os.listdir(data_path[files_count])
        for file in files:
            image = cv2.imread(data_path[files_count]+'/'+file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)   # 在灰度图中检测面部
            print("检测到的人脸个数: {}".format(len(rects)))
            for k, rect in enumerate(rects):

                #得到LBP纹理图
                # height , width  = gray.shape
                # img_lbp = np.zeros((height, width),
                #                    np.uint8)
                # for i in range(0, height):
                #     for j in range(0, width):
                #         img_lbp[i, j] = lbp_calculated_pixel(gray, i, j)


                # 绘制得到特征的点数据
                shape = predictor(gray, rect)# 提取出体征点
                # for pt in shape.parts():
                #     pt_pos = (pt.x, pt.y)
                #     cv2.circle(img_lbp, pt_pos, 2, 255, -1)
                for pt in shape.parts():
                    if pt.x < 48 and pt.y < 48:
                        gray[[pt.y], [pt.x]] = 255
                    else:pass
                for x, y, z, w in rect:
                    roiImg = gray[y:y + w, x:x + z]
                # height = 48
                # width = 48
                # img_last = np.zeros((height, width), np.uint8)
                # for i in range(height):
                #     for j in range(width):
                #         if img_lbp[i][j] == 255:
                #             img_last[i][j] = img_lbp[i][j]
                roiImg = cv2.resize(roiImg,(256,256))
                cv2.imshow('img_last',roiImg)  ####最终图片展示
                cv2.waitKey(0)
                # plt.imshow(img_last, cmap="gray")
                # plt.show()
                cv2.imwrite(filename=output+'/'+file,img = roiImg)
            # data.append(img_last)####直接保留图片矩阵进行编码
            # data.append(np.ndarray.flatten(img_last))  ###展开一维数据
            # label.append(emotion_bit[files_count])
        # write_to_file(data,label)

def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value

# Function for calculating Jaffe_feature
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val



if __name__ == '__main__':
    image_pre()
