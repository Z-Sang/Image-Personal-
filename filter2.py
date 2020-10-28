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

def clampling(inten):
    if(inten > 255):
        inten = 255
    elif(inten < 0):
        inten = 0
    else:
        inten = inten
    return inten

def zeropadding(img,number):
    height,width = img.shape
    padding_img = np.array([[0]*(width+2*number)]*(height+2*number))
    for i in range(height):
        for j in range(width):
            padding_img[i+number][j+number] = img[i][j]
    return padding_img

def filter_image(img,coef):
    height,width = img.shape
    filter_img = np.array([[0]*width]*height)
    num = len(coef[0])
    if num == 3 :
        img = zeropadding(img,1)
        sum_coef = coef[0][0] + coef[0][1] + coef[0][2] + coef[1][0] + coef[1][1] + coef[1][2] + coef[2][0] + coef[2][1] + coef[2][2]     
        if sum_coef == 0:
            sum_coef = 1
        for i in range(height):
            for j in range(width):
                filter_img[i][j] = 1/sum_coef * (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                        + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                        + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
                filter_img[i][j] = clampling(filter_img[i][j])
    if num == 5 :
        img = zeropadding(img,2)
        sum_coef = coef[0][0] + coef[0][1] + coef[0][2] + coef[0][3] + coef[0][4] + coef[1][0] + coef[1][1] + coef[1][2] + coef[1][3] + coef[1][4] + coef[2][0] + coef[2][1] + coef[2][2] + coef[2][3] + coef[2][4] + coef[3][0] + coef[3][1] + coef[3][2] + coef[3][3] + coef[3][4] + coef[4][0] + coef[4][1] + coef[4][2] + coef[4][3] + coef[4][4]
        for i in range(height):
            for j in range(width):
                filter_img[i][j] = 1/sum_coef * (img[i-2][j-2] * coef[0][0] + img[i-2][j-1] * coef[0][1] + img[i-2][j] * coef[0][2] + img[i-2][j+1] * coef[0][3] + img[i-2][j+2] * coef[0][4] 
                                                + img[i-1][j-2] * coef[1][0] + img[i-1][j-1] * coef[1][1] + img[i-1][j] * coef[1][2] + img[i-1][j+1] * coef[1][3] + img[i-1][j+2] * coef[1][4]
                                                + img[i][j-2] * coef[2][0] + img[i][j-1] * coef[2][1] + img[i][j] * coef[2][2] + img[i][j+1] * coef[2][3] + img[i][j+2] * coef[2][4] 
                                                + img[i+1][j-2] * coef[3][0] + img[i+1][j-1] * coef[3][1] + img[i+1][j] * coef[3][2] + img[i+1][j+1] * coef[3][3] + img[i+1][j+2] * coef[3][4] 
                                                + img[i+2][j-2] * coef[4][0] + img[i+2][j-1] * coef[4][1] + img[i+2][j] * coef[4][2] + img[i+2][j+1] * coef[4][3] + img[i+2][j+2] * coef[4][4]) 
                filter_img[i][j] = clampling(filter_img[i][j])
    return filter_img

def maximum_filter(img,num):
    height,width = img.shape
    maximum_img = np.array([[0]*width]*height)
    maximum = np.array([0]*(num*num))
    if num == 3 :
        img = zeropadding(img,1)
        for i in range(height):
            for j in range(width):
                maximum = [img[i-1][j-1],img[i-1][j],img[i-1][j+1],img[i][j-1],img[i][j],img[i][j+1],img[i+1][j-1],img[i+1][j],img[i+1][j+1]]
                maximum_img[i][j] = np.max(maximum)
    if num == 5 :
        img = zeropadding(img,2)
        for i in range(height):
            for j in range(width):
                maximum = [img[i-2][j-2],img[i-2][j-1],img[i-2][j],img[i-2][j+1],img[i-2][j+2],
                           img[i-1][j-2],img[i-2][j-1],img[i-2][j],img[i-2][j+1],img[i-2][j+2],
                           img[i][j-2],img[i][j-1],img[i][j],img[i][j+1],img[i][j+2],
                           img[i+1][j-2],img[i+1][j-1],img[i+1][j],img[i-2][j+1],img[i+1][j+2],
                           img[i+2][j-2],img[i+2][j-1],img[i+2][j],img[i-2][j+1],img[i+2][j+2]]
                maximum_img[i][j] = np.max(maximum)    
    return maximum_img

def minimum_filter(img,num):
    height,width = img.shape
    minimum_img = np.array([[0]*width]*height)
    minimum = np.array([0]*(num*num))
    if num == 3 :
        img = zeropadding(img,1)
        for i in range(height):
            for j in range(width):
                minimum = [img[i-1][j-1],img[i-1][j],img[i-1][j+1],img[i][j-1],img[i][j],img[i][j+1],img[i+1][j-1],img[i+1][j],img[i+1][j+1]]
                minimum_img[i][j] = np.min(minimum)
    if num == 5 :
        img = zeropadding(img,2)
        for i in range(height):
            for j in range(width):
                minimum = [img[i-2][j-2],img[i-2][j-1],img[i-2][j],img[i-2][j+1],img[i-2][j+2],
                           img[i-1][j-2],img[i-2][j-1],img[i-2][j],img[i-2][j+1],img[i-2][j+2],
                           img[i][j-2],img[i][j-1],img[i][j],img[i][j+1],img[i][j+2],
                           img[i+1][j-2],img[i+1][j-1],img[i+1][j],img[i-2][j+1],img[i+1][j+2],
                           img[i+2][j-2],img[i+2][j-1],img[i+2][j],img[i-2][j+1],img[i+2][j+2]]
                minimum_img[i][j] = np.min(minimum)    
    return minimum_img

def median_filter(img,weight_median):
    height,width = img.shape
    num = len(weight_median[0])
    median_img = np.array([[0]*width]*height)
    sum_weight = 0
    for i in range(num):
        for j in range(num):
            sum_weight = sum_weight + weight_median[i][j]
    median = np.array([0]*sum_weight)
    if num == 3 :
        img = zeropadding(img,1)
        for i in range(height):
            for j in range(width):
                count = 0
                for a in range(num):
                    for b in range(num):
                        for c in range(weight_median[a][b]):
                            median[count] = img[i-1+a][j-1+b]
                            count = count+1
                median_img[i][j] = np.median(median)                
    if num == 5 :
        img = zeropadding(img,2)
        for i in range(height):
            for j in range(width):
                count = 0
                for a in range(num):
                    for b in range(num):
                        for c in range(weight_median[a][b]):
                            median[count] = img[i-2+a][j-2+b]
                            count = count+1
                median_img[i][j] = np.median(median)      

    return median_img

img = cv2.imread('tree.jpg',0)

## define coefficient

weight = [[1,1,1],[1,2,1],[1,1,1]]
weight2 = [[1,1,1,1,1],[1,2,2,2,1],[1,2,2,2,1],[1,2,2,2,1],[1,1,1,1,1]]

median_coef = [[1,1,1],[1,1,1],[1,1,1]]
median_coef2 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]

weight_median = [[1,2,1],[2,3,2],[1,2,1]]
weight_median2 = [[1,2,4,2,1],[2,4,8,4,2],[4,8,16,8,4],[2,4,8,4,2],[1,2,4,2,1]]

gaussian = [[1,2,1],[2,4,2],[1,2,1]]
gaussian2 = [[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]

img2 = np.array(filter_image(img,weight),dtype='uint8')
img3 = np.array(filter_image(img,weight2),dtype='uint8')
img4 = np.array(filter_image(img,gaussian),dtype='uint8')
img5 = np.array(filter_image(img,gaussian2),dtype='uint8')
img6 = np.array(maximum_filter(img,3),dtype='uint8')
img7 = np.array(maximum_filter(img,5),dtype='uint8')
img8 = np.array(minimum_filter(img,3),dtype='uint8')
img9 = np.array(minimum_filter(img,5),dtype='uint8')
img10 = np.array(median_filter(img,median_coef),dtype='uint8')
img11 = np.array(median_filter(img,weight_median),dtype='uint8')
img12 = np.array(median_filter(img,median_coef2),dtype='uint8')
img13 = np.array(median_filter(img,weight_median2),dtype='uint8')

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.imshow('img3',img3)
cv2.waitKey(0)
cv2.imshow('img4',img4)
cv2.waitKey(0)
cv2.imshow('img5',img5)
cv2.waitKey(0)
cv2.imshow('img6',img6)
cv2.waitKey(0)
cv2.imshow('img7',img7)
cv2.waitKey(0)
cv2.imshow('img8',img8)
cv2.waitKey(0)
cv2.imshow('img9',img9)
cv2.waitKey(0)
cv2.imshow('img10',img10)
cv2.waitKey(0)
cv2.imshow('img11',img11)
cv2.waitKey(0)
cv2.imshow('img12',img12)
cv2.waitKey(0)
cv2.imshow('img13',img13)
cv2.waitKey(0)