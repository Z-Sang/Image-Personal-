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
                filter_img[i][j] = (img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                    + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            elif i == height-1 and j == width-1: 
                filter_img[i][j] = (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] 
                                    + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1])
            elif i == 0 and j == width-1: 
                filter_img[i][j] = (img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] 
                                    + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1])
            elif i == height-1 and j == 0: 
                filter_img[i][j] = (img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                    + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2])
            elif i == 0:
                filter_img[i][j] = (img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                    + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            elif j == 0: 
                filter_img[i][j] = (img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                    + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                    + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
            elif i == height-1:
                filter_img[i][j] = (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                    + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2])
            elif j == width-1: 
                filter_img[i][j] = (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1]  
                                    + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] 
                                    + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1])
            else:
                filter_img[i][j] = (img[i-1][j-1] * coef[0][0] + img[i-1][j] * coef[0][1] + img[i-1][j+1] * coef[0][2] 
                                    + img[i][j-1] * coef[1][0] + img[i][j] * coef[1][1] + img[i][j+1] * coef[1][2]
                                    + img[i+1][j-1] * coef[2][0] + img[i+1][j] * coef[2][1] + img[i+1][j+1] * coef[2][2])
    
    return filter_img

img = cv2.imread('tree.jpg',0)
histogram = calculateHistogram(img)
box = [[1,1,1],[1,2,1],[1,1,1]]
gaussian = [[1,2,1],[2,4,2],[1,2,1]]
maxican_hat = [[0,-1,0],[-1,8,-1],[0,-1,0]]
img2 =  np.array(filter_image(img,box)/10,dtype='uint8')
img3 =  np.array(filter_image(img,gaussian)/16,dtype='uint8')
img4 =  np.array(filter_image(img,maxican_hat)/4,dtype='uint8')
histogram2 = calculateHistogram(img2)
histogram3 = calculateHistogram(img3)
histogram4 = calculateHistogram(img4)
cv2.imshow('img',img)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.imshow('img4',img4)
cv2.waitKey(0)

# fig, axs = plt.subplots(2,2) 
# axs[0,0].bar(np.linspace(0,255,num = 256),histogram,width = 1)
# axs[0,0].set_title('Histogram of Original')
# axs[0,1].bar(np.linspace(0,255,num = 256),histogram2,width = 1)
# axs[0,1].set_title('Histogram of Box filter')
# axs[1,0].bar(np.linspace(0,255,num = 256),histogram3,width = 1)
# axs[1,0].set_title('Histogram of Gaussian filter')
# axs[1,1].bar(np.linspace(0,255,num = 256),histogram4,width = 1)
# axs[1,1].set_title('Histogram of Maxican hat filter')
# plt.show()

# fig, (ax1,ax2) = plt.subplots(2,1) 
# ax1.bar(np.linspace(0,255,num = 256),histogram,width = 1)
# ax1.set_title('Histogram of Original')
# ax2.bar(np.linspace(0,255,num = 256),histogram2,width = 1)
# ax2.set_title('Histogram of Box filter')

# fig2, (ax1,ax2) = plt.subplots(2,1) 
# ax1.bar(np.linspace(0,255,num = 256),histogram,width = 1)
# ax1.set_title('Histogram of Original')
# ax2.bar(np.linspace(0,255,num = 256),histogram3,width = 1)
# ax2.set_title('Histogram of Gaussian filter')

# fig3, (ax1,ax2) = plt.subplots(2,1) 
# ax1.bar(np.linspace(0,255,num = 256),histogram,width = 1)
# ax1.set_title('Histogram of Original')
# ax2.bar(np.linspace(0,255,num = 256),histogram4,width = 1)
# ax2.set_title('Histogram of Maxican hat filter')
# plt.show()