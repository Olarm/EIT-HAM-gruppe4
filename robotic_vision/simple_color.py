import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

#dir = "/Users/ola/dev/eit/data/signDatabasePublicFramesOnly/vid6/frameAnnotations-MVI_0071.MOV_annotations/"
dir = "/Users/ola/dev/eit/data/stop_signs/"

def find_contours(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_square(img, contours):
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (np.sqrt(w**2 + h**2) > 5) and (np.sqrt(w**2 - h**2) < 50):
            cv2.rectangle(img ,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),2)

    return img

def get_mask(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = np.array([161, 145, 60])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv, low_red, high_red)

    low_red2 = np.array([0, 155, 60])
    high_red2 = np.array([25, 255, 255])
    red_mask2 = cv2.inRange(hsv, low_red2, high_red2)

    return red_mask + red_mask2

def main():
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            img_file = os.path.join(dir, filename)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            mask = get_mask(img)
            contours = find_contours(mask)
            img_rect = draw_square(img, contours)

            cv2.imshow(filename, img_rect)
            key = cv2.waitKey(0)
            if key == 27:  # (escape to quit)
                sys.exit()



main()