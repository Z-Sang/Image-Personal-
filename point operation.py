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

def adjust_contrast(image,value_to_adjust):
    contrast = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            contrast[i][j] = image[i][j] * value_to_adjust
            contrast[i][j] = clampling(contrast[i][j])
    return contrast

def adjust_brightness(image,value_to_adjust):
    brightness = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            brightness[i][j] = image[i][j] + value_to_adjust
            brightness[i][j] = clampling(brightness[i][j])
    return brightness

def inverse_image(image):
    inverse = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            inverse[i][j] = 255 - image[i][j]
    return inverse

def clampling(inten):
    if(inten > 255):
        inten = 255
    elif(inten < 0):
        inten = 0
    else:
        inten = inten
    return inten

img = cv2.imread('tree.jpg',0)
height,width = img.shape

contrast_img = adjust_contrast(img,1.5)
brightness_img = adjust_brightness(img,10)
inverse_img = inverse_image(img)

intensity = calculateHistogram(img)
intensity_contrast = calculateHistogram(contrast_img)
intensity_brightness = calculateHistogram(brightness_img)
intensity_inverse = calculateHistogram(inverse_img)

img2 = np.array(inverse_img,dtype='uint8')
cv2.imshow("image",img2)
cv2.waitKey(0) 

fig, (ax1,ax2) = plt.subplots(2,1) 
ax1.bar(np.linspace(0,255,num = 256),intensity)
ax1.set_title('Original')
# ax2.bar(np.linspace(0,255,num = 256),intensity_contrast)
# ax2.set_title('Contrast')
# ax2.bar(np.linspace(0,255,num = 256),intensity_brightness)
# ax2.set_title('Brightness')
ax2.bar(np.linspace(0,255,num = 256),intensity_inverse)
ax2.set_title('Inverse')
plt.show()
