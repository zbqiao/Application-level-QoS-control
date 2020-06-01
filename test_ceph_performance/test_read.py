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
time_sum  = 0.0
cnt = 0
size = range(25)
for i in size:
    name = str(i)+"test"
    mb = ioctx_2.stat(name)[0]/1024/1024
    print "Actual size = ",mb
    a=time.time()
    delta_L0_L1_str = ioctx_2.read(name,ioctx_2.stat(name)[0],0)
    b =time.time()
    #print "Read data size = 32 MB", 
    print "Read time =", b-a
    
    time_sum += b-a
    cnt+=1
    #time.sleep(1)
print "Average time = ",time_sum/cnt
print "Accumulate time = ",time_sum
