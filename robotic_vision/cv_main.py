import numpy as np
import matplotlib.pyplot as plt
import cv2

filename       = '../data/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/stop_1323804473.avi_image13.png'

img = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


### Canny edge detection
threshold1 = 200
threshold2 = 200

canny = cv2.Canny(gray, threshold1, threshold2) 

### Hough transform
hough_lines = cv2.HoughLines(canny, 1, np.pi/180, 100) 
img_hough = np.zeros_like(img)

for x in range(0, len(hough_lines)):
    for rho, theta in hough_lines[x]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img_hough,(x1,y1),(x2,y2),(0,0,255),2)

### Convert images from BGR to RGB
RGB_img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
RGB_canny = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
RGB_hough = cv2.cvtColor(img_hough, cv2.COLOR_BGR2RGB)


### Plot
fig, axes = plt.subplots(1, 3, figsize=[15,8])
axes[0].imshow(RGB_img)
axes[1].imshow(RGB_canny)
axes[2].imshow(RGB_hough)
plt.show()
