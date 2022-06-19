import time

import cv2
import numpy as np
from PIL import Image

from lmsn import LMSN

if __name__ == "__main__":
    lmsn = LMSN()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the mode of the test:
    #   'predict' means single picture prediction
    #   'video' means video detection
    #   'fps' means test fps
    #   'dir_predict' means traverse the folder to detect and save
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"

    #-------------------------------------------------------------------------#
    #   test_interval is used to specify the number of image detections when measuring fps
    #-------------------------------------------------------------------------#
    test_interval = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path of the image used for detection
    #   dir_save_path specifies the save path of the detected image
    #-------------------------------------------------------------------------#
    dir_origin_path = "img_RSOD/"
    dir_save_path   = "img_out_RSOD/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = lmsn.detect_image(image)
                r_image.show()

    elif mode == "fps":
        img = Image.open('img_RSOD/aircraft_4.jpg')
        tact_time = lmsn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = lmsn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
