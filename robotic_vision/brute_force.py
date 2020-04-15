import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('../data/templates/stop_template2.jpg')
img2 = cv.imread('../data/test/stopsign/16.jpg')


gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv.ORB_create(nfeatures=2000)
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
# Match descriptors.
#bf.add(np.array(des1))
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print(matches)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('ORB', img3)
cv.waitKey()
