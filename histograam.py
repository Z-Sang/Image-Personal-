import cv2 
import numpy as np
from matplotlib import pyplot as plt

min_intensity = 0
max_intensity = 256

img = cv2.imread('8_bit.png')
blueChannel = img[:,:,0]
greenChannel = img[:,:,1]
redChannel = img[:,:,2]
# print(img.shape)
# cv2.imshow("image",img)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256]); plt.show()

def histogram(channel):
    for i in list_img[channel]:
        intensity[channel][int(i)] +=1

    for i in range(max_intensity):
        if(i == 0):
            cumulative[channel][i] = intensity[channel][i]
        else:
            cumulative[channel][i] = intensity[channel][i] + cumulative[channel][i-1]


list_img = [[] for i in range(3)]
intensity = [[0 for i in range(max_intensity)] for i in range(3)]
cumulative = [[0 for i in range(max_intensity)] for i in range(3)]

for row,index in enumerate(blueChannel):
    # print(row)
    list_img[0] = np.concatenate((list_img[0],index))

for row,index in enumerate(greenChannel):
    # print(row)
    list_img[1] = np.concatenate((list_img[1],index))

for row,index in enumerate(redChannel):
    # print(row)
    list_img[2] = np.concatenate((list_img[2],index))

histogram(0)
histogram(1)
histogram(2)
# print(len(list_img))
# print(cumulative[255])
# histr = cv2.calcHist([img],[2],None,[max_intensity],[min_intensity,max_intensity])
plt.figure("Histogram")
plt.subplot(2,1,1)
plt.title("No Tool")
plt.plot(intensity[0],color = 'b')
plt.plot(intensity[1],color = 'g')
plt.plot(intensity[2],color = 'r')
plt.subplot(2,1,2)
plt.title("From CV")
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.figure("Cumulative")
plt.subplot(2,1,1)
plt.title("Histogram")
plt.plot(intensity[0],color = 'b')
plt.plot(intensity[1],color = 'g')
plt.plot(intensity[2],color = 'r')
plt.subplot(2,1,2)
plt.title("Cumulative Histogram")
plt.plot(cumulative[0],color = 'b')
plt.plot(cumulative[1],color = 'g')
plt.plot(cumulative[2],color = 'r')
plt.show()

img2 = cv2.imread('8_bit.png',0)
# print(img.shape)
# cv2.imshow("image",img)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256]); plt.show()

list_img2 = []
intensity2 = [0 for i in range(max_intensity)]
cumulative2 = [0 for i in range(max_intensity)]

for row,index in enumerate(img2):
    # print(row)
    list_img2 = np.concatenate((list_img2,index))

for i in list_img2:
    intensity2[int(i)] +=1

for i in range(max_intensity):
    if(i == 0):
        cumulative2[i] = intensity2[i]
    else:
        cumulative2[i] = intensity2[i] + cumulative2[i-1]
# print(len(list_img))
# print(cumulative[255])
histr2 = cv2.calcHist([img2],[0],None,[max_intensity],[min_intensity,max_intensity])

plt.figure("Histogram")
plt.subplot(2,1,1)
plt.title("No Tool")
plt.plot(intensity2)
plt.subplot(2,1,2)
plt.title("From CV")
plt.plot(histr2)
plt.figure("Cumulative")
plt.subplot(2,1,1)
plt.title("Histogram")
plt.plot(intensity2)
plt.subplot(2,1,2)
plt.title("Cumulative Histogram")
plt.plot(cumulative2)
plt.show()