import cv2 
import numpy as np
from matplotlib import pyplot as plt

min_intensity = 0
max_intensity = 256

def calculateHistogram(img):
    list_img = []
    for row,index in enumerate(img):
        list_img = np.concatenate((list_img,index))
    intensity = [0 for i in range(max_intensity)]
    for i in list_img:
        intensity[int(i)] +=1
    return intensity

def calculateCdf(img):
    intensity = calculateHistogram(img)
    array_intensity = np.array(intensity)
    cdf = array_intensity.cumsum()
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf

def histogram_specification(img_target,img_input):
    intensity_img = calculateHistogram(img_target)
    intensity_img2 = calculateHistogram(img_input)

    height,width = img_input.shape
    specification = np.array([[0]*width]*height)

    cumulative_img = calculateCdf(img_target)
    cumulative_img2 = calculateCdf(img_input)

    index = [0 for i in range(max_intensity)]

    for i in range(max_intensity):
        for j in range(max_intensity):
            if j == 255:
                index[i] = j
                break
            elif np.abs(cumulative_img2[i]-cumulative_img[j]) < np.abs(cumulative_img2[i]-cumulative_img[j+1]):
                index[i] = j
                break
    
    for i in range(height):
        for j in range(width):
            specification[i][j] = index[img2[i][j]]  
    return specification              

img = cv2.imread('talay.jpg',0)
cumulative = calculateCdf(img)
histogram = calculateHistogram(img)
img2 = cv2.imread('talay2.jpg',0)
cumulative2 = calculateCdf(img2)
histogram2 = calculateHistogram(img2)
img3 = np.array(histogram_specification(img,img2),dtype='uint8')
cumulative3 = calculateCdf(img3)
histogram3 = calculateHistogram(img3)

# cv2.imshow('target',img)
# cv2.imshow('input',img2)
# cv2.imshow('output',img3)
# cv2.waitKey(0)

fig, (ax1,ax2,ax3) = plt.subplots(3,1) 
ax1.bar(np.linspace(0,255,num = 256),cumulative)
ax1.set_title('Cumulative Histogram of Target')
ax2.bar(np.linspace(0,255,num = 256),cumulative2)
ax2.set_title('Cumulative Histogram of Input')
ax3.bar(np.linspace(0,255,num = 256),cumulative3)
ax3.set_title('Cumulative Histogram of Output')
plt.show()

fig2, (ax1,ax2,ax3) = plt.subplots(3,1) 
ax1.bar(np.linspace(0,255,num = 256),histogram)
ax1.set_title('Histogram of Target')
ax2.bar(np.linspace(0,255,num = 256),histogram2)
ax2.set_title('Histogram of Input')
ax3.bar(np.linspace(0,255,num = 256),histogram3)
ax3.set_title('Histogram of Output')
plt.show()