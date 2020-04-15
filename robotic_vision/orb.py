import cv2
import numpy as np
import matplotlib.pyplot as plt


def orb(template, img):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = [i for i in matches if i.distance < 60]
    
    if len(matches) >= 5:
        img3 = cv2.drawMatches(template,kp1,img,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img3
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

    return matches


filename = '/Users/ola/dev/eit/data/valid/stopsign/48.jpg'
template = '/Users/ola/dev/eit/data/templates/stop_template2.jpg'
img_m = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
template_m = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

blurred_mask = cv2.blur(red_mask, (3,3))
ret, thresh = cv2.threshold(red_mask, 127, 255, 0)
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = []
for cnt in contours:
    [x,y,w,h] = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt) 

    #if (np.square(w**2 - h**2) < 50) and (area > 100):
    if area > 200:
        print(area)
        cropped = img_m[y:y+h, x:x+w]
        matches = orb(template_m, img_m)

        cv2.rectangle(img_m, (x, y), (x+w, y+h),(0,0,255),2)

#plt.imshow(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB))
plt.show()