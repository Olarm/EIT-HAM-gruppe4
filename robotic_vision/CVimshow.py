import cv2
import numpy as np
from matplotlib import pyplot as plt

from simple_color import *

filename = '/Users/ola/dev/eit/data/signDatabasePublicFramesOnly/vid6/frameAnnotations-MVI_0071.MOV_annotations/stop_1323896946.avi_image13.png'
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img_g = cv2.imread(filename)
img_b = cv2.blur(img, (3,3))


color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img_g],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
#mask = get_mask(img_b)
#contours = find_contours(mask)
#img_sq = draw_square(img, contours)

#cv2.imshow("test", img_b)
#cv2.waitKey(0)
