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

def plot(data,r,z,filename):
    points = np.transpose(np.array([z,r]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    fig,ax=plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=26)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)

    axis_font = {'fontname':'Arial', 'size':'38'}

    #plt.xlabel('R', **axis_font)
    #plt.ylabel('Z',**axis_font )
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=25));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig(filename, format='png')

reduced_len = 5145
filename = "reduced_data.bin"
f = open(filename, "rb")
dpot_L1_str=f.read(reduced_len*8)
#dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
r_L1_str=f.read(reduced_len*8)
#r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
z_L1_str=f.read(reduced_len*8)
#z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
f.close()
dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)

start = time.time()
plot(dpot_L1,r_L1,z_L1,"astro2d/test2/reduced.png")
end = time.time()
print "Plot time = ", end -start


