from __future__ import division
import rados
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import triangulation
from scipy.spatial import Delaunay
from scipy import stats
import numpy as np
import math
import time
import struct
import sys
import os
import subprocess
from sklearn.cluster import KMeans
#from scipy.fftpack import fft
import scipy.fftpack as fft
import Queue
from threading import Thread
import zfpy
import argparse
import cv2
from scipy.spatial import Delaunay
from skimage.measure import compare_ssim
#from skimage.measure import structural_similarity as ssim
import imutils
np.set_printoptions(threshold=np.inf)

#image.append(cv2.imread("test.png"))
image_original = cv2.imread("astro2d/original_no_upsampling.png",cv2.IMREAD_GRAYSCALE)
image_reduced = cv2.imread("astro2d/8X.png",cv2.IMREAD_GRAYSCALE)
#image_reduced = cv2.imread("",cv2.IMREAD_GRAYSCALE)

#image_original = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
#image_reduced = cv2.cvtColor(image_red, cv2.COLOR_BGR2GRAY)
image_original = image_original/255
image_reduced = image_reduced/255
union = image_original * image_reduced
dice = 2*np.sum(union)/(np.sum(image_original)+np.sum(image_reduced))
print "Dice=", dice



image_original = cv2.imread("astro2d/original_no_upsampling.png")
image_reduced = cv2.imread("astro2d/8X.png")
gray_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
gray_reduced = cv2.cvtColor(image_reduced, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(gray_original, gray_reduced, full=True)
diff = (diff * 255).astype("uint8")

print("SSIM: {}".format(score))
 



#image_original = image_original
#image_reduced = image_reduced
#cnt=0
#for i in range(np.shape(image_original)[0]):
#    for j in range(np.shape(image_original)[1]):
#        if image_original[i][j] == image_reduced[i][j]:
#            cnt+=1
#print 2*cnt/(np.shape(image_original)[0]*np.shape(image_original)[1] + np.shape(image_reduced)[0]*np.shape(image_reduced)[1])       
#print np.shape(image_original)
#print np.shape(image_reduced)
#union = image_original * image_reduced
#print np.shape(union)
#dice = 2*np.sum(union)/(np.sum(image_original)+np.sum(image_reduced))
#print dice
#s = ssim(image_original, image_reduced)
#Convert the images to grayscale
#gray_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
#gray_reduced = cv2.cvtColor(image_reduced, cv2.COLOR_BGR2GRAY)

#(score, diff) = compare_ssim(gray_original, gray_reduced, full=True)
#print diff
#diff = (diff * 255).astype("uint8")
#print diff

#print("SSIM: {}".format(score))
#print "s=",s


 
