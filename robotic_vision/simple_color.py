import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

#dir = "/Users/ola/dev/eit/data/signDatabasePublicFramesOnly/vid6/frameAnnotations-MVI_0071.MOV_annotations/"
dir = "/Users/ola/dev/eit/data/stop_signs/"

def find_contours(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def filter_contours(contours):
    bounds = [[]]
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (np.sqrt(w**2 + h**2) > 5) and (np.sqrt(w**2 - h**2) < 50):
            bounds.append([x, y, w, h])


def draw_square(img, contours):
    print("########################################")
    print("#############  NEXT IMAGE  #############")
    print("########################################")
    bounds = np.empty([0,4], dtype=np.int)
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (np.sqrt(w**2 + h**2) > 3):# and (np.sqrt(w**2 - h**2) < 50):
            bounds = np.append(bounds, [[x, y, w, h]], axis=0)

    H, W = bounds.shape
    merged = np.empty([0])
    for i, b in enumerate(bounds):
         for j, k in enumerate(bounds):
            if not ((i in merged) or (j in merged)):
                distance = np.sqrt((b[0] - k[0])**2 + (b[1] - k[1])**2 + (b[2] - k[2])**2 + (b[3] - k[3])**2)
                if (distance > 0) and (distance < 30):
                    temp = np.append([b], [k], axis=0)
                    print(temp)
                    x = np.amin([b[0], k[0]])
                    y = np.amin([b[1], k[1]])
                    w = max(b[2], k[2])
                    h = b[3] + k[3]
                    bounds = np.append(bounds, [[x,y,w,h]], axis=0)
                    merged = np.append(merged, i)
                    merged = np.append(merged, j)

    merged = np.unique(merged)
    bounds = np.delete(bounds, merged, 0)

    for b in bounds:
        cv2.rectangle(img ,(b[0]-5,b[1]-5),(b[0]+b[2]+5,b[1]+b[3]+5),(0,0,255),2)
    return img

def get_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    low_red1 = np.array([161, 145, 20])
    high_red1 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv, low_red1, high_red1)

    low_red2 = np.array([0, 145, 50])
    high_red2 = np.array([21, 255, 255])
    red_mask2 = cv2.inRange(hsv, low_red2, high_red2)

    low_red3 = np.array([161, 20, 125])
    high_red3 = np.array([179, 155, 255])
    red_mask3 = cv2.inRange(hls, low_red3, high_red3)

    low_red4 = np.array([0, 50, 125])
    high_red4 = np.array([21, 155, 255])
    red_mask4 = cv2.inRange(hls, low_red4, high_red4)

    red_mask = (red_mask1 + red_mask2 + red_mask3 + red_mask4)

    return red_mask

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