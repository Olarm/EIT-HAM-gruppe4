import numpy as np
import cv2 as cv


def orb_match(template, img):

    orb = cv.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)


    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    #bf.add(des1)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    img3 = cv.drawMatches(template,kp1,img,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img3
