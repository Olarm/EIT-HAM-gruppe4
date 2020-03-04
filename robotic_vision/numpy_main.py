#                              README
# This script expects you to have filled out the functions declared
# in common1.py: blur, central_difference and extract_edges. If you
# define these according to the expected input and output, you should
# be able to simply run this file to generate the figure for task 1.
#
import numpy as np
import matplotlib.pyplot as plt
from methods import *
from PIL import Image

edge_threshold = 0.02# todo: choose an appropriate value
blur_sigma     = 0.5# todo: choose an appropriate value
filename       = '../data/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/stop_1323804473.avi_image13.png'

I_png      = Image.open(filename)
I_rgb      = I_png.convert("RGB")
I_rgb.save('colors.jpg')
I_rgb      = plt.imread("colors.jpg")
I_rgb      = I_rgb/255.0
I_gray     = rgb2gray(I_rgb)
I_blur     = blur(I_gray, blur_sigma)
Iu, Iv, Im = central_difference(I_blur)
u,v,theta  = extract_edges(Iu, Iv, Im, edge_threshold)
supp_edges = non_max_suppression(Iu, Iv, Im, edge_threshold)


fig, axes = plt.subplots(2,3,figsize=[15,10], sharey='row', sharex='row')
plt.set_cmap('gray')
axes[0, 0].imshow(I_blur)
axes[0, 1].imshow(Iu, vmin=-0.05, vmax=0.05)
axes[0, 2].imshow(Iv, vmin=-0.05, vmax=0.05)
axes[1, 0].imshow(Im, vmin=+0.00, vmax=0.10, interpolation='bilinear')
edges = axes[1, 1].scatter(u, v, s=1, c=theta, cmap='hsv')
fig.colorbar(edges, ax=axes[1, 1], orientation='horizontal', label='$\\theta$ (radians)')
for row in axes:
    for col in row:
        col.set_xlim([0, I_rgb.shape[1]])
        col.set_ylim([I_rgb.shape[0], 0])
        col.set_aspect('equal')
axes[1, 2].imshow(supp_edges, vmin=-0.05, vmax=0.05)
axes[0, 0].set_title('Blurred input')
axes[0, 1].set_title('Gradient in u')
axes[0, 2].set_title('Gradient in v')
axes[1, 0].set_title('Gradient magnitude')
axes[1, 1].set_title('Extracted edges')
axes[1, 2].set_title('suppressed edges')
plt.tight_layout()
plt.savefig('out1.png')
plt.show()
