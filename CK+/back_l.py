import cv2
cascade = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")
def face_rec():
    gray = cv2.imread('ori_data/contempt/S138_008_00000007.png')
    save_path = 'feature_bit_data/'
    rect = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=9, minSize=(50, 50))
    print("rect", rect)
    if not rect is ():
        for x, y, z, w in rect:
            roiImg = gray[y:y + w, x:x + z]
            cv2.rectangle(gray, (x, y), (x + z, y + w), (0, 0, 255), 2)
            cv2.imshow('frame', gray)
            cv2.waitKey()
            roiImg = cv2.resize(roiImg, (256, 256))
            cv2.imwrite(save_path + str(1) + '.jpg', roiImg)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    face_rec()