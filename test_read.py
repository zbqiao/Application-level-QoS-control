# -*- coding: utf-8 -*-
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
np.set_printoptions(threshold=np.inf)

try:
    cluster = rados.Rados(conffile='')
except TypeError as e:
    print 'Argument validation error: ', e
    raise e

try:
    cluster.connect()
except Exception as e:
    print "connection error: ", e
    raise e
if not cluster.pool_exists('tier2_pool'):
    raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('tier2_pool')
print "start\n"
read_tag = 1
sum_bw = 0
bw = []
size = [1, 2,4,8,16,32,64,128,256,512,1024]
if read_tag == 1:
    for i in size:
        mb = ioctx_2.stat(str(i))[0]/1024/1024
        print "Actual size = ",mb
        a=time.time()
        delta_L0_L1_str = ioctx_2.read(str(i),ioctx_2.stat(str(i))[0],0)
        b =time.time()
        print "Read data size = ", i
        #print "Read time =", b-a
        print "bandwidth =", mb/(b-a)
        bw.append(mb/(b-a))
        print "**********************************************"
    #delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",ioctx_2.stat("delta_r_L0_L1_o")[0],0)
    #delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",ioctx_2.stat("delta_z_L0_L1_o")[0],0)
else:
    for i in range(10):
        a=time.time()
        delta_L0_L1_str = ioctx_2.read("delta_L0_L1_o",1,0)
        b =time.time()
        print "One time read =", b-a
    #delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",1,0)
    #delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",1,0)
print "Bandwidth = ",bw
#data_str = ioctx_2.read("data_o",ioctx_2.stat("data_o")[0],0)
#r_str = ioctx_2.read("r_o",ioctx_2.stat("r_o")[0],0)
#z_str = ioctx_2.read("z_o",ioctx_2.stat("z_o")[0],0)

#data = struct.unpack(str(int(ioctx_2.stat("data_o")[0]/8))+'d',data_str)
#r = struct.unpack(str(int(ioctx_2.stat("r_o")[0]/8))+'d',r_str)
#z = struct.unpack(str(int(ioctx_2.stat("z_o")[0]/8))+'d',z_str)

#points = np.transpose(np.array([z,r]))
#print "Start\n"
#a = time.time()
#Delaunay_t = Delaunay(points)
#b = time.time()
#print "End\n"
#print b-a
