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
#finally:
#    print "Connected to the cluster."

if not cluster.pool_exists('test_hdd'):
	raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('test_hdd')
bytesize = []
#max 512MB
for i in range(11):
    bytesize.append(1024*1024*2**i)
bytesize.insert(0,bytesize[0])
read_time = []
for i in bytesize:
    print i/1024/1024
    #bytesize.append(1024*1024*i)
    #range(0,256*1024*1024, )
    start = time.time()
    #temp_str = ioctx_2.read(str(int(i/1024/1024)),i,0)
    temp_str = ioctx_2.read("all_diff",i,0)
    end = time.time()
    read_time.append(end-start)
print "Byte size=", bytesize
print "Time = ", read_time
ioctx_2.close()
cluster.shutdown()
