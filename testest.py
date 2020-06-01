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
filename = "reduced_data.bin"

f = open(filename, "rb")
dpot_L1_compressed=f.read(4325048*8)
dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
r_L1_compressed=f.read(2975952*8)
r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
z_L1_compressed=f.read(2516984*8)
z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
f.close()
#base_gradient=np.gradient(dpot_L1)
#
#print base_gradient
#sorted_gradient = sorted(base_gradient)
#sorted_index = np.argsort(base_gradient)
#interval = len(base_gradient)//10
#remain = len(base_gradient)% 10
#blocklist=range(0,len(base_gradient),interval)
#blocklist[-1] = blocklist[-1] + remain
#
#for i in range(len(blocklist)-1):
#    tag = 0.1*i    
#    delta = delta_L0_L1[blocklist[i]:blocklist[i+1]]
#    delta_r
print "dpot_L1[0]=",dpot_L1[0]
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

if not cluster.pool_exists('tier2_pool'):
	raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('tier2_pool')
block_num = 10
block_interval = 1/block_num
delta_index_L0_L1_str = ""
delta_L0_L1_str = ""
delta_r_L0_L1_str = ""
delta_z_L0_L1_str = ""
delta_len=0

delta_read_time = 0.0
for i in range(int(block_num)):
	print i*block_interval
	chosn_delta_name= "delta_L0_L1_"+str(i*block_interval)
	chosn_delta_r_name = "delta_r_L0_L1_"+str(i*block_interval)
	chosn_delta_z_name = "delta_z_L0_L1_"+str(i*block_interval)
	chosn_index_name = "delta_index_" + str(i*block_interval)
	#print chosn_delta_name, chosn_delta_r_name, chosn_delta_z_name, chosn_index_name
	start = time.time()
	delta_index_L0_L1_str += ioctx_2.read(chosn_index_name,ioctx_2.stat(chosn_index_name)[0],0)
	delta_L0_L1_str += ioctx_2.read(chosn_delta_name,ioctx_2.stat(chosn_delta_name)[0],0)
	delta_r_L0_L1_str += ioctx_2.read(chosn_delta_r_name,ioctx_2.stat(chosn_delta_r_name)[0],0)
	delta_z_L0_L1_str += ioctx_2.read(chosn_delta_z_name,ioctx_2.stat(chosn_delta_z_name)[0],0)
	end = time.time()
	#print ioctx_2.stat(chosn_index_name)[0]/4, ioctx_2.stat(chosn_delta_name)[0]/8, ioctx_2.stat(chosn_delta_r_name)[0]/8, ioctx_2.stat(chosn_delta_z_name)[0]/8
	delta_read_time+= end-start
	delta_len += int(ioctx_2.stat(chosn_delta_name)[0]/8)

print "read time=",delta_read_time

start1=time.time()
delta_L0_L1_str1 = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
delta_r_L0_L1_str1= ioctx_2.read("delta_r_L0_L1_o",ioctx_2.stat("delta_r_L0_L1_o")[0],0)
delta_z_L0_L1_str1 = ioctx_2.read("delta_z_L0_L1_o",ioctx_2.stat("delta_z_L0_L1_o")[0],0)
end1 = time.time()
delta_read_time1 = end1-start1
print "Delta reading time = ", delta_read_time1
ioctx_2.close()
cluster.shutdown()

