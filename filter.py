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