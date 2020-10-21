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

def filter_image(img,coef):
    height,width = img.shape
    filter_img = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                filter_img[i][j] = 1/9*(img[i][j] * coef[0][0] + img[i][j] * coef[0][1] + img[i][j+1] * coef[0][2] 
                                        + img[i][j] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                        + img[i+1][j] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            elif i == 0:
                filter_img[i][j] = 1/9*(img[i][j-1] * coef[0][0] + img[i][j] * coef[0][1] + img[i][j+1] * coef[0][2] 
                                        + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                        + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            elif j == 0: 
                filter_img[i][j] = 1/9*(img[i-1][j] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                        + img[i][j] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                        + img[i+1][j] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            else:
                filter_img[i][j] = 1/9*(img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                        + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                        + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
    
    return filter_img

img = cv2.imread('talay.jpg',0)
histogram = calculateHistogram(img)
img2 = cv2.imread('talay.jpg',0)
histogram2 = calculateHistogram(img2)
coef = [[0,0,0],[0,1,0],[0,0,0]]
img3 =  np.array(filter_image(img2,coef),dtype='uint8_t')
cv2.imshow('output',img3)
cv2.waitKey(0)