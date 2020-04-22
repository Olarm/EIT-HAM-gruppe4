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


def cnt_angles(contour):
    temp_contour = np.empty((1,2))
    temp_contour[0] = contour[-1]
    temp_contour = np.append(temp_contour, contour[:, 0], axis=0)
    temp_contour = np.append(temp_contour, contour[0], axis=0)
    angles = np.empty((contour.shape[0], 1))
    for i, cnt in enumerate(contour):
        vec1 = [temp_contour[i+1,0] - temp_contour[i,0], temp_contour[i,1] - temp_contour[i+1,1]]
        vec2 = [temp_contour[i+1,0] - temp_contour[i+2,0], temp_contour[i+2,1] - temp_contour[i+1,1]]
        vec1 = vec1 / np.sqrt(vec1[0]**2 + vec1[1]**2)
        vec2 = vec2 / np.sqrt(vec2[0]**2 + vec2[1]**2)

        angle1 = np.arctan2(vec1[0], vec1[1])
        angle2 = np.arctan2(vec2[0], vec2[1])

        #angle = angle2 - angle1
        angles[i] = angle2 - angle1

        #if (angle < -2.40 and angle > -2.60) or (angle < 4.03 and angle > 3.82):
        #    count += 1 

    return angles

def hough_lines(img_gray):
    try:
        edges = cv.Canny(img_gray,50,150,apertureSize = 3)
        minLineLength = 5
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
        return edges, lines
    except:
        print("Canny error")
        return None, None


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
            #cv2.rectangle(img ,(x,y),(x+w,y+h),(0,0,255), 2)
            cv2.drawContours(img, contours, i, (0,0,255), 1)
            matches = orb_detect(img_gray[y:y+h, x:x+w])
            hull_1 = cv2.convexHull(cnt, returnPoints=False)
            hull = cv2.convexHull(cnt)
            angles = cnt_angles(hull)
            cv2.drawContours(img, hull, -1, (255,0,0), 4)
            defects = cv2.convexityDefects(cnt, hull_1)
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            hull_approx = cv2.approxPolyDP(hull, 0.009 * cv2.arcLength(hull, True), True)
            if len(angles) >= 4:
                temp_angles = np.empty(1)
                temp_hull = np.empty((1,2))
                for j, k in enumerate(hull[:,0]):
                    angle = angles[j][0]
                    if ((angle < -2.3 and angle > -2.6) or (angle > 3.2 and angle < 3.8)):
                        temp_angles = np.append(temp_angles, angle)
                        temp_hull = np.append(temp_hull, k, axis=0)
                if len(temp_angles) >= 4:
                    print(temp_hull)
                    for j, k in enumerate(temp_hull):
                        cv.putText(img, str(angle[j])[0:4], (k[0], k[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if len(matches) > 0:
                cv.putText(img, str(len(matches)), (x-5, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(hull_approx) == 8:
                cv.putText(img, str(len(hull_approx)), (x-5, y+h+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if len(matches) >= 5:
                np.append(new_contours, contours[i])
                cv.putText(img, "Stop sign", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif len(approx) == 8 and len(defects) == 0:
                np.append(new_contours, contours[i])
                cv.putText(img, "Stop sign", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif len(approx) == 8 and len(defects) > 0:
                cv.putText(img, str(len(defects)), (x+w+5, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



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
            
            [img_detected, reduced_contours] = detect(img_gray, img, contours)
            #img_rect = draw_square(img_detected, reduced_contours)

            #img_resized = cv2.resize(img_detected, (960, 540))
            cv2.imshow(filename, img_detected)
            key = cv2.waitKey(0)
            if key == 27:  # (escape to quit)
                sys.exit()
            
            #elif key == 119:
                #cv.imwrite("detected_"+filename, img_detected)



main()