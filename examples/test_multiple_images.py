import sys
sys.path.append('../keras-yolo3/')

import yolo
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from timeit import default_timer as timer

def detect_img(yolo, images_path):
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    stop_flag = False
    
    while True:
        for image_path in images_path:
            try:
                image = Image.open(image_path)
            except:
                print('Open Error! Try again!')
            else:
                image = yolo.detect_image(image)
                result = np.asarray(image)
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                k = cv2.waitKey(33)
                if k == 27:    # Esc key to stop
                    stop_flag = True
                    break
                elif k == -1:  # normally -1 returned,so don't print it
                    continue
                else:
                    print (k) # else print its value
        if(stop_flag is True):
            break
    cv2.destroyAllWindows()

def close_yolo(yolo):
    yolo.close_session()
    
if __name__ == '__main__':
    image_path = list()
    for i in range(100):
        image_path.append('D:/_videos/MOT2017/train/MOT17-02-DPM/img1/%s.jpg'%(str(i).zfill(6)))
    
    detect_img(yolo.YOLO(), image_path)