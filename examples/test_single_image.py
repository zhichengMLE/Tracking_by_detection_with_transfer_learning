import sys
sys.path.append('../keras-yolo3/')

import yolo
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np

def detect_img(yolo, images_path):
    for image_path in images_path:
        try:
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(image)
            result = np.asarray(r_image)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def close_yolo(yolo):
    yolo.close_session()
    
if __name__ == '__main__':
    image_path = ['D:/_videos/MOT2017/train/MOT17-02-DPM/img1/000001.jpg']
    
    detect_img(yolo.YOLO(), image_path)