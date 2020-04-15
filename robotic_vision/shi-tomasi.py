import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/templates/stop_template1.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=50,      qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)

for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
#cv2.imshow('good_features', img)
#cv2.waitKey()