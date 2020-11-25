import cv2 
import numpy as np
from matplotlib import pyplot as plt

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
    img = zeropadding(img,1)
    sum_coef = abs(coef[0][0]) +  abs(coef[0][1]) +  abs(coef[0][2]) +  abs(coef[1][0]) +  abs(coef[1][1]) +  abs(coef[1][2]) +  abs(coef[2][0]) +  abs(coef[2][1]) +  abs(coef[2][2])     
    if sum_coef == 0:
        sum_coef = 1
    for i in range(height):
        for j in range(width):
            filter_img[i][j] = 1/sum_coef * (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            filter_img[i][j] = clampling(filter_img[i][j])
        
    return filter_img

def  calculateGradient_magnitude(img_horizontal,img_vertical):
    height,width = img.shape
    magnitude_img = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            magnitude_img[i][j] = np.sqrt(np.power(img_horizontal[i][j],2) + np.power(img_vertical[i][j],2))
            magnitude_img[i][j] = clampling(magnitude_img[i][j])

    return magnitude_img

def  calculateGradient_orientation(img_horizontal,img_vertical):
    height,width = img.shape
    orientation_img = np.array([[0]*width]*height)
    for i in range(height):
        for j in range(width):
            orientation_img[i][j] = np.arctan2(img_vertical[i][j],img_horizontal[i][j]) * 180 / np.pi 
            orientation_img[i][j] = clampling(orientation_img[i][j])

    return orientation_img

img = cv2.imread('tree.jpg',0)

prewitt_horizontal = [[-1,0,1],[-1,0,1],[-1,0,1]]
prewitt_vertical = [[-1,-1,-1],[0,0,0],[1,1,1]]
sobel_horizontal = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobel_vertical = [[-1,-2,-1],[0,0,0],[1,2,1]]
modified_sobel_horizontal = [[-3,0,3],[-10,0,10],[-3,0,3]]
modified_sobel_vertical = [[-3,-10,-3],[0,0,0],[3,10,3]]

img2 = filter_image(img,prewitt_horizontal)
img3 = filter_image(img,prewitt_vertical)
img4 = filter_image(img,sobel_horizontal)
img5 = filter_image(img,sobel_vertical)
img6 = filter_image(img,modified_sobel_horizontal)
img7 = filter_image(img,modified_sobel_vertical)

img8 = np.array(calculateGradient_magnitude(img2,img3),dtype='uint8')
img9 = np.array(calculateGradient_magnitude(img4,img5),dtype='uint8')
img10 = np.array(calculateGradient_magnitude(img6,img7),dtype='uint8')

img11 = np.array(calculateGradient_orientation(img2,img3),dtype='uint8')
img12 = np.array(calculateGradient_orientation(img4,img5),dtype='uint8')
img13 = np.array(calculateGradient_orientation(img6,img7),dtype='uint8')

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imshow('img2',np.array(img2,dtype ='uint8'))
cv2.waitKey(0)
cv2.imshow('img3',np.array(img3,dtype ='uint8'))
cv2.waitKey(0)
cv2.imshow('img4',np.array(img4,dtype ='uint8'))
cv2.waitKey(0)
cv2.imshow('img5',np.array(img5,dtype ='uint8'))
cv2.waitKey(0)
cv2.imshow('img6',np.array(img6,dtype ='uint8'))
cv2.waitKey(0)
cv2.imshow('img7',np.array(img7,dtype ='uint8'))
cv2.waitKey(0)

cv2.imshow('img8',img8)
cv2.waitKey(0)
cv2.imshow('img11',img11)
cv2.waitKey(0)

cv2.imshow('img9',img9)
cv2.waitKey(0)
cv2.imshow('img12',img12)
cv2.waitKey(0)

cv2.imshow('img10',img10)
cv2.waitKey(0)
cv2.imshow('img13',img13)
cv2.waitKey(0)