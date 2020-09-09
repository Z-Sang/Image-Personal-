import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('8_bit.png',0)
print(img.shape)
cv2.imshow("image",img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
plt.hist(img.ravel(),256,[0,256]); plt.show()
list_img = []
for row,index in enumerate(img):
    # print(row)
    list_img = np.concatenate((list_img,index))
print(len(list_img))
color = ('b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
