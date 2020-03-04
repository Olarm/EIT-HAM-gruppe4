import numpy as np
from scipy.stats import norm
from scipy.ndimage import convolve

# Task 1a
def central_difference(I):
    """
    Computes the gradient in the u and v direction using
    a central difference filter, and returns the resulting
    gradient images (Iu, Iv) and the gradient magnitude Im.
    """

    s = I.shape
    p = np.pad(I, 1, mode="constant")
    c = np.array([0.5, 0, -0.5])

    Iu = np.zeros_like(I) # Placeholder
    Iv = np.zeros_like(I) # Placeholder
    Im = np.zeros_like(I) # Placeholder

    for v in range(0, s[0]):
        for u in range(0, s[1]):
            iu = p[v+1, u:u+3]
            iv = p[v:v+3, u+1]
            Iv[v, u] = np.convolve(iv, c, "valid")
            Iu[v, u] = np.convolve(iu, c, "valid")
            Im[v, u] = np.sqrt(Iu[v, u]**2 + Iv[v, u]**2)
    return Iu, Iv, Im

# Task 1b
def blur(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel, w, should depend on sigma, e.g.
    # w=2*np.ceil(3*sigma) + 1. Also, ensure that the blurred image
    # has the same size as the input image.

    padding = int(np.ceil(3 * sigma))
    padded = np.pad(I, padding, mode="constant")
    width = 2 * padding + 1
    x = np.linspace(-sigma, sigma, width + 1)
    gaus1d = np.diff(norm.cdf(x))
    gaus2d = np.outer(gaus1d, gaus1d)
    gaus2d = gaus2d / gaus2d.sum()
    gaus2d = np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]]) / 159

    result = np.zeros_like(I) # Placeholder

    for v in range(0, I.shape[0]):
        for u in range(0, I.shape[1]):
            f = padded[v:v+width, u:u+width]
            result[v, u] = np.sum(gaus2d * f)

    return result

# Task 1c
def extract_edges(Iu, Iv, Im, threshold):
    """
    Returns the u and v coordinates of pixels whose gradient
    magnitude is greater than the threshold.
    """

    # This is an acceptable solution for the task (you don't
    # need to do anything here). However, it results in thick
    # edges. If you want better results you can try to replace
    # this with a thinning algorithm as described in the text.
    v,u = np.nonzero(Im > threshold)
    theta = np.arctan2(Iv[v,u], Iu[v,u])
    return u, v, theta

def non_max_suppression(Iu, Iv, Im, threshold):
    edges = np.zeros_like(Im) - 255
    v,u = np.nonzero(Im > threshold)
    theta = np.arctan2(Iv[v,u], Iu[v,u])
    theta = theta * 180 / np.pi

    Im_p = np.pad(Im, 1, mode="constant")

    for i in range(u.shape[0]):
        #angle 0
        if (-22.5 <= theta[i] < 22.5) or (157.5 <= theta[i]) or (theta[i] < -157.5):
             q = Im_p[v[i]+1, u[i]+2]
             r = Im_p[v[i]+1, u[i]]
        #angle 45
        elif (22.5 <= theta[i] < 67.5) or (-112.5 > theta[i] >= -157.5):
            q = Im_p[v[i]+2, u[i]+2]
            r = Im_p[v[i], u[i]]
        #angle 90
        elif (67.5 <= theta[i] < 112.5) or (-112.5 <= theta[i] < -67.5):
            q = Im_p[v[i]+2, u[i]+1]
            r = Im_p[v[i], u[i]+1]
        #angle 135
        elif (112.5 <= theta[i] < 157.5) or (-67.5 <= theta[i] < -22.5):
            q = Im_p[v[i], u[i]]
            r = Im_p[v[i]+2, u[i]+2]

        if (Im[v[i],u[i]] >= q) and (Im[v[i],u[i]] >= r):
            edges[v[i],u[i]] = Im[v[i],u[i]]
        else:
            edges[v[i],u[i]] = -255

    return edges



def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]
