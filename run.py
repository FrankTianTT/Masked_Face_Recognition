import cv2
import retinex
import os
import glob

if __name__ == '__main__':
    DATA_ROOT = './train/0000045'
    for name in os.listdir(DATA_ROOT):
        img_path = os.path.join(DATA_ROOT, name)
        sigma = [15, 80, 200]
        print(img_path)
        img = cv2.imread(img_path)
        img_en = retinex.automatedMSRCR(img, sigma)

        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 300, 300)
        cv2.moveWindow("img", 100, 100)
        cv2.namedWindow("img_en", 0)
        cv2.resizeWindow("img_en", 300, 300)
        cv2.moveWindow("img_en", 400, 100)

        cv2.imshow('img', img)
        cv2.imshow('img_en', img_en)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
