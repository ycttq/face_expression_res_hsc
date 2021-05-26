# -- coding: utf-8 --
from nupic.algorithms.spatial_pooler import SpatialPooler as SP
from nupic.algorithms.knn_classifier import KNNClassifier
import dlib
import cv2
import numpy as np
import csv
import time
from matplotlib import pyplot as plt
emotion_bit = ['anger','contempt','disgust','fear','happy','sadness','surprise']

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

sp = SP(inputDimensions=(1, 2304),

        columnDimensions=(1, 2304),

        potentialRadius=2304,

        potentialPct=0.85,

        globalInhibition=True,

        localAreaDensity=-1,

        numActiveColumnsPerInhArea=240,

        stimulusThreshold=0,

        synPermInactiveDec=0,

        synPermActiveInc=0,

        synPermConnected=0.2,

        minPctOverlapDutyCycle=0.001,

        dutyCyclePeriod=1000,

        boostStrength=0.0,

        seed=1956)

classifier = KNNClassifier(k=5, distanceNorm=2)
def train():
    with open('train_data/face_all_bit_train.csv', 'r') as train_file:
       reader = csv.reader(train_file)
       for train_count, record in enumerate(reader):
          train = np.array(record[1:2305])
          column = np.zeros((2304))
          sp.compute(train, False, column)
          classifier.learn(column, record[0])
    train_file.close()
    print '分类器训练完成'
def face_test(face_data_last):
    test = np.array(face_data_last)
    column = np.zeros((2304))
    sp.compute(test,False,column)
    winner = classifier.infer(column)[0]
    return emotion_bit[winner]
def Non_zero_to_0ne(data_list): #定义编码方式
    bit = []
    for i in data_list:
        if i == 0:
            bit.append(0)
        else:
            bit.append(1)
    return bit
def get_pixel(img, center, x, y): #LBP算法过程
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
def lbp_calculated_pixel(img, x, y):##进行脸部归一化
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
def face_rec(): ### 实时人脸图像采集
    save_path = 'src/'
    cascade = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=9, minSize=(50, 50))
        print("rect", rect)
        if not rect is ():
            for x, y, z, w in rect:
                roiImg = gray[y:y + w, x:x + z]
                cv2.rectangle(frame, (x, y), (x + z, y + w), (0, 0, 255), 2)
                i += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            roiImg = cv2.resize(roiImg, (48, 48))
            cv2.imwrite(save_path + str(i) + '.jpg', roiImg)
            break
    cap.release()
    cv2.destroyAllWindows()
    return roiImg ,frame

def face_data_extra(gray):  ##表情识别
    rects = detector(gray, 1)  # 在灰度图中检测面部
    print("检测到的人脸个数: {}".format(len(rects)))
    height, width = gray.shape
    img_lbp = np.zeros((height, width),
                       np.uint8)
    for k, rect in enumerate(rects):

        # 得到LBP纹理图
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(gray, i, j)

        # 绘制得到特征的点数据
        shape = predictor(gray, rect)  # 提取出体征点
        for pt in shape.parts():
            if pt.x < 48 and pt.y < 48:
                img_lbp[[pt.y], [pt.x]] = 255
            else:
                pass
        # cv2.imwrite('last_face.png',img_lbp)
    face_data = Non_zero_to_0ne(np.ndarray.flatten(img_lbp))
    return face_data
if __name__ == '__main__':
    train()
    while True:
        face_yuanshi,frame = face_rec()
        face_data_last = face_data_extra(face_yuanshi)
        emotion_result = face_test(face_data_last)
        print '当前的表情为' + emotion_result
        cv2.putText(frame, emotion_result, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv2.imshow('Facial expression recognition ',frame)
        time.sleep(5)