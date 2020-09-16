import cv2 
import numpy as np
from matplotlib import pyplot as plt

min_intensity = 0
max_intensity = 256

def calculateHistogram(img):
    list_img = []
    intensity = [0 for i in range(max_intensity)]
    for row,index in enumerate(img):
        list_img = np.concatenate((list_img,index))

    for i in list_img:
        intensity[int(i)] +=1

    return intensity

def calculateCumulativeHistogram(intensity):
    cumulative = [0 for i in range(max_intensity)]
    for i in range(max_intensity):
        if(i == 0):
            cumulative[i] = intensity[i]
        else:
            cumulative[i] = (intensity[i] + cumulative[i-1])

    return cumulative

img = cv2.imread('tree.jpg',0)

cv2.imshow("image",img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

intensity = calculateHistogram(img)
cumulative = calculateCumulativeHistogram(intensity)
histr = cv2.calcHist([img],[0],None,[max_intensity],[min_intensity,max_intensity])
cumulat = histr.cumsum()
histogram = [0]*max_intensity
for i in range(max_intensity):
    histogram[i] = int(histr[i])

plt.figure("Histogram")
plt.subplot(2,1,1)
plt.title("From CV")
plt.bar(np.linspace(0,255,num = 256),histogram)
plt.subplot(2,1,2)
plt.title("No Tool")
plt.bar(np.linspace(0,255,num = 256),intensity)
plt.figure("Cumulative")
plt.subplot(2,1,1)
plt.title("Cumulative Histogram")
plt.bar(np.linspace(0,255,num = 256),cumulat)
plt.subplot(2,1,2)
plt.title("Cumulative Histogram No Tool")
plt.bar(np.linspace(0,255,num = 256),cumulative)
plt.show()