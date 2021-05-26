from matplotlib import pyplot as plt
import dlib
from PIL import Image
import cv2
import numpy as np
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

predictor_path = 'shape-predictor/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# ima_paths = ['ori_data/anger/S010_004_00000017.png','ori_data/disgust/S005_001_00000009.png','ori_data/contempt/S148_002_00000013.png','ori_data/fear/S502_004_00000050.png'
#             ,'ori_data/happy/S011_006_00000013.png','ori_data/sadness/S014_002_00000016.png','ori_data/surprise/S022_001_00000029.png']
# img_1 = Image.open('ori_data/anger/S010_004_00000017.png')
# img_2 = Image.open('ori_data/disgust/S005_001_00000009.png')
# img_3 = Image.open('ori_data/contempt/S148_002_00000013.png')
# img_4 = Image.open('ori_data/fear/S502_004_00000050.png')
# img_5 = Image.open('ori_data/happy/S011_006_00000013.png')
# img_6 = Image.open('ori_data/sadness/S014_002_00000016.png')
# img_7 = Image.open('ori_data/surprise/S022_001_00000029.png')
num = 0
ima_paths = ['ori_data/sadness/S014_002_00000016.png']

image_lsit = []
for ima_path in ima_paths:
    image = cv2.imread(ima_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)   # 在灰度图中检测面部
    print("检测到的人脸个数: {}".format(len(rects)))
    for k, rect in enumerate(rects):

        # 得到LBP纹理图
        height, width = gray.shape
        img_lbp = np.zeros((height, width),
                           np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(gray, i, j)

        # 绘制得到特征的点数据
        shape = predictor(gray, rect)  # 提取出体征点
        for pt in shape.parts():
            if pt.x < 48 and pt.y < 48:
                gray[[pt.y], [pt.x]] = 255
            else:
                pass
        image_lsit.append(image)
        image_lsit.append(gray)
        image_lsit.append(img_lbp)
emotion = ['anger','anger','anger','disgust','disgust','disgust','contempt','contempt','contempt','fear','fear','fear','happy','happy','happy','sadness','sadness','sadness','surprise','surprise','surprise']
emotion_2 = ['sadness','sadness','sadness']
for i in range(3):#### 展示图数据

    ax=plt.subplot(1 , 3, i + 1)
    plt.imshow(image_lsit[i],cmap='gray')
    ax.set_title(emotion_2[i])
    plt.subplots_adjust(hspace=None,wspace=None)
    plt.axis('off')
    plt.tight_layout
plt.show()