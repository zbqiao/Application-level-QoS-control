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

if not cluster.pool_exists('tier2_pool'):
    raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('tier2_pool')

float a[3];
print sys.getsizeof(a) 

ioctx_2.close()
cluster.shutdown()
