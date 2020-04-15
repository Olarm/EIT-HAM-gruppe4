import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

from detectors import *

template = "/Users/ola/dev/eit/data/templates/stop_template2.jpg"

#dir = "/Users/ola/dev/eit/data/signDatabasePublicFramesOnly/vid6/frameAnnotations-MVI_0071.MOV_annotations/"
#dir = "/Users/ola/dev/eit/data/stop_signs/"
dir = "/Users/ola/dev/eit/data/test/stopsign/"
#dir = "/Users/ola/dev/eit/data/valid/stopsign/"



def orb_detect(img_gray):
    template_img = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    try:
        if not des2:
            return []
    except:
        pass

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = [i for i in matches if i.distance < 60]
    
    return matches

    if len(matches) >= 5:
        return 
        #img3 = cv2.drawMatches(template,kp1,img,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #return img3
    #plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

    return False


def detect(img_gray, img, contours):
    new_contours = []
    for i, cnt in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(cnt)
        if np.sqrt(w**2 + h**2) > 10 and h > 1 and w > 1:
            cv2.rectangle(img ,(x,y),(x+w,y+h),(0,0,255),2)
            matches = orb_detect(img_gray[y:y+h, x:x+w])
            if len(matches) > 0:
                cv.putText(img, str(len(matches)), (x-5, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if len(matches) >= 5:
                np.append(new_contours, contours[i])
                cv.putText(img, "Stop sign", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #print("#### OLD SHAPE: ", contours.shape)
    #print("#### NEW SHAPE: ", new_contours.shape)
    return [img, new_contours]


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
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        cv2.rectangle(img ,(y-5,x-5),(y+h+5,x+w+5),(0,0,255),2)
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

    red_hsv = red_mask1
    red_hsv[np.where(red_mask2 > 0)] = 255

    red_hls = red_mask3
    red_hls[np.where(red_mask4 > 0)] = 255

    red_mask = red_hsv
    red_mask[np.where(red_hls > 0)] = 255


    low_white = np.array([0, 0, 0])
    high_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, low_white, high_white)

    lower_gray = np.array([0, 5, 50], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(img, img, mask = mask_gray)

    rgb = img/255.0 # Convert to floating-point to prevent overflow
    # Compute distance in BGR space to strong red
    difference = rgb - np.array([0.0,0.0,1.0])
    distance = np.linalg.norm(difference, axis=2) # Euclidean length (L2 norm)of the third dimension (rgb difference)
    thresholded = distance < 0.7 # Isolate pixels that are sufficiently close to strong red

    red_mask = red_mask1
    red_mask[np.where(red_mask2 > 0)] = 255
    red_mask[np.where(red_mask3 > 0)] = 255
    red_mask[np.where(red_mask4 > 0)] = 255
    red_mask[np.where(thresholded > 0)] = 255

    return red_mask

def main():

    template_img = cv2.imread(template, cv2.IMREAD_COLOR)

    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_file = os.path.join(dir, filename)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = get_mask(img)
            blurred_mask = cv2.blur(mask, (5,5))
            contours = find_contours(mask)

            #gray = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
            edges = cv.Canny(blurred_mask,50,150,apertureSize = 3)
            lines = cv.HoughLines(edges,1,np.pi/180,200)
            
            try:
                for line in lines:
                    rho,theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    #cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            except TypeError:
                print("typeerror #######")

            [img_detected, reduced_contours] = detect(img_gray, img, contours)
            #img_rect = draw_square(img_detected, contours)

            #img_resized = cv2.resize(img_detected, (960, 540))
            cv2.imshow(filename, img_detected)
            key = cv2.waitKey(0)
            if key == 27:  # (escape to quit)
                sys.exit()



main()