import cv2 
import numpy as np
from matplotlib import pyplot as plt

min_intensity = 0
max_intensity = 256

img = cv2.imread('8_bit.png',0)
# print(img.shape)
# cv2.imshow("image",img)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256]); plt.show()

list_img = []
intensity = [0 for i in range(max_intensity)]

for row,index in enumerate(img):
    # print(row)
    list_img = np.concatenate((list_img,index))

for i in list_img:
    intensity[int(i)] +=1

histr = cv2.calcHist([img],[0],None,[max_intensity],[min_intensity,max_intensity])

plt.figure("Histogram")
plt.subplot(2,1,1)
plt.title("No Tool")
plt.plot(intensity)
plt.subplot(2,1,2)
plt.title("From CV")
plt.plot(histr)
plt.show()