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

img = cv2.imread('treelowcontrast2.jpg',0)
img2 = cv2.imread('tree.jpg',0)
img3 = cv2.imread('treecontrast.jpg',0)
# cv2.imshow("image",img)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# intensity = calculateHistogram(img)
# cumulative = calculateCumulativeHistogram(intensity)
# histr = cv2.calcHist([img],[0],None,[max_intensity],[min_intensity,max_intensity])
# cumulat = histr.cumsum()
hist1 =  cv2.calcHist([img],[0],None,[max_intensity],[min_intensity,max_intensity])
hist2 =  cv2.calcHist([img2],[0],None,[max_intensity],[min_intensity,max_intensity])
hist3 =  cv2.calcHist([img3],[0],None,[max_intensity],[min_intensity,max_intensity])

# plt.figure("Histogram")
# plt.subplot(2,1,1)
# plt.title("From CV")
# plt.plot(histr)
# plt.subplot(2,1,2)
# plt.title("No Tool")
# plt.plot(intensity)
# plt.figure("Cumulative")
# plt.subplot(2,1,1)
# plt.title("Cumulative Histogram")
# plt.plot(cumulat)
# plt.subplot(2,1,2)
# plt.title("Cumulative Histogram No Tool")
# plt.plot(cumulative)
# plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.plot(hist1)
ax1.set_title('Low Contrast')
ax2.plot(hist2)
ax2.set_title('Normal Contrast')
ax3.plot(hist3)
ax3.set_title('High Contrast')
plt.show()
