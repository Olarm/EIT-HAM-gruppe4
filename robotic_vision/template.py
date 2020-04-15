import cv2
import imutils
import numpy as np 
   
# Read the main image 
img_rgb = cv2.imread('../data/test/stopsign/16.jpg')
   
# Convert to grayscale 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

   
# Read the template 
template = cv2.imread('../data/templates/stop_template2.jpg', 0)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
cv2.waitKey(0) 