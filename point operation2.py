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

def threshold(image,thres):
    threshold = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            if(image[i][j] < thres):
                threshold[i][j] = 0
            elif(image[i][j] >= thres):
                threshold[i][j] = 255
    return threshold

def automatic_contrast(image,a_min,a_max):
    auto_contrast = np.array([[0]*width]*height)
    a_low = 255
    a_high = 0
    for i in range(height):
        for j in range(width):
            if(a_low > image[i][j]):
                a_low = image[i][j]
            if(a_high < image[i][j]):
                a_high = image[i][j]
    for i in range(height):
        for j in range(width):
            auto_contrast[i][j] = a_min + (image[i][j] - a_low) * (a_max-a_min)/(a_high-a_low)
    return auto_contrast

def calculateCumulativeHistogram(intensity):
    cumulative = [0 for i in range(max_intensity)]
    for i in range(max_intensity):
        if(i == 0):
            cumulative[i] = intensity[i]
        else:
            cumulative[i] = (intensity[i] + cumulative[i-1])
    return cumulative

def calculate_low(cumulative,qlow):
    for i in range(max_intensity):
        if(cumulative[i]>=width*height*qlow):
            return i

def calculate_high(cumulative,qhigh):
    for i in range(max_intensity):
        if(cumulative[i]<=width*height*(1-qhigh)):
            a_max = i
    return a_max

def modified_contrast(image,a_low,a_high,a_min,a_max):
    modified = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            if(image[i][j]<= a_low):
                modified[i][j] = a_min
            elif(image[i][j] > a_low and image[i][j] < a_high):
                modified[i][j] = a_min + (image[i][j] - a_low) * (a_max - a_min)/(a_high - a_low)
            elif(image[i][j] >= a_high):
                modified[i][j] = a_max
    return modified

def calculateMaxIntensity(intensity):
    max_inten = []
    for i in range(max_intensity):
        if(intensity[i] != 0):
            max_inten.append(i)
    return max(max_inten)

def histogram_equalization(image,cumulative,max_inten):
    equalization = np.array([[0]*width]*height)
    total_pixel = width*height
    index = []
    for i in range(max_intensity):
        index.append(round((cumulative[i]/total_pixel)*max_inten))
    for i in range(height):
        for j in range(width):
            equalization[i][j] = index[image[i][j]]  
    return equalization                  

img = cv2.imread('tree.jpg',0)
height,width = img.shape

threshold_img = threshold(img,100)
auto_contrast_img = automatic_contrast(img,0,255)

intensity = calculateHistogram(img)
intensity_threshold = calculateHistogram(threshold_img)
intensity_auto_contrast = calculateHistogram(auto_contrast_img)

cumulative = calculateCumulativeHistogram(intensity)
a_low = calculate_low(cumulative,0.005)
a_high = calculate_high(cumulative,0.005)

modified_contrast_img = modified_contrast(img,a_low,a_high,0,255)
intensity_modified_contrast = calculateHistogram(modified_contrast_img)

max_inten = calculateMaxIntensity(intensity)
histogram_equalization_img = histogram_equalization(img,cumulative,max_inten)
intensity_histogram_equal = calculateHistogram(histogram_equalization_img)
cumulative_histogram_equal = calculateCumulativeHistogram(intensity_histogram_equal)

img2 = np.array(histogram_equalization_img,dtype='uint8')
cv2.imshow("image",img2)
cv2.waitKey(0) 

fig, (ax1,ax2) = plt.subplots(2,1) 
ax1.bar(np.linspace(0,255,num = 256),intensity)
ax1.set_title('Original')
# ax2.bar(np.linspace(0,255,num = 256),intensity_threshold)
# ax2.set_title('Threshold')
# ax2.bar(np.linspace(0,255,num = 256),intensity_auto_contrast)
# ax2.set_title('Auto Contrast')
# ax2.bar(np.linspace(0,255,num = 256),intensity_modified_contrast)
# ax2.set_title('Modified Contrast')
ax2.bar(np.linspace(0,255,num = 256),intensity_histogram_equal)
ax2.set_title('Histogram Equalization')
plt.show()

fig, (ax1,ax2) = plt.subplots(2,1) 
ax1.bar(np.linspace(0,255,num = 256),cumulative)
ax1.set_title('Cumulative Histogram of Original')
ax2.bar(np.linspace(0,255,num = 256),cumulative_histogram_equal)
ax2.set_title('Cumulative Histogram of Histogram Equalization')
plt.show()