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

def pdist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

def blob_detection(fname):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image")
    args = vars(ap.parse_args())

    # load the image
    #image = cv2.imread(args["image"])
    image = []
    #image.append(cv2.imread('/home/qliu/Dropbox/DataReductionEvaluate-2018-restored/figures/mergeCompress/dpot-1x.png'))
    name_hat = "/media/sf_shared/"
    name_tail1 = "original"
    #name_tail2 = "original"
    name_tail2 = "finer_0"
    print (name_tail2)
    name1 = name_hat+name_tail1+".png"
    name2 = name_hat+name_tail2+".png"
    image.append(cv2.imread("original.png"))
    image.append(cv2.imread(fname))
    #image.append(cv2.imread("test.png"))
    #image.append(cv2.imread("xgca_256_stats_preserving.png"))
    #image.append(cv2.imread("xgca_8.png"))
    #image.append(cv2.imread("dpot-16.png"))
    #image.append(cv2.imread("dpot-32.png"))
    #image.append(cv2.imread("dpot-64.png"))

    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c4-zfp.png"))
    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c4-sz.png"))
    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c5-zfp.png"))
    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c5-sz.png"))
    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c6-zfp.png"))
    #image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c6-sz.png"))

    # define the list of boundaries
    boundaries = [
        ([0, 0, 100], [204, 204, 255]), #red 
        ([86, 31, 4], [220, 88, 50]),
        ([25, 146, 190], [62, 174, 250]),
        ([103, 86, 65], [145, 133, 128])
    ]

    (lower, upper) = boundaries[0]
    print (boundaries[0])

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask


    # show the images
    #plt.imshow(output)
    #plt.show()

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;
    print (params.thresholdStep)
    # Filter by Area.
    params.filterByArea = 1
    params.minArea = 120

    # Filter by Circularity
    #params.filterByCircularity = True
    #params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = 1
    params.minConvexity = 0.3
    #params.maxConvexity = 1


    # Filter by Inertia
    params.filterByInertia = 1
    params.minInertiaRatio = 0.1
    #params.maxInertiaRatio = 1

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)


    mask = []
    output = []
    gray_image = []
    keypoints = []
    abc = cv2.inRange(image[0], lower, upper)
    print(len(image))
    for i in range(len(image)):
        print i
        time1 = time.time()
        mask.append(cv2.inRange(image[i], lower, upper))
        output.append(cv2.bitwise_and(image[i], image[i], mask = mask[i]))
        gray_image.append(cv2.cvtColor(output[i], cv2.COLOR_BGR2GRAY))
        keypoints.append(detector.detect(gray_image[i]))
        time2 = time.time()
        print ('blob detection time', (time2 - time1))
        print ('image %d blob # %d' %(i, len(keypoints[i])))


        total_diameter = 0
        total_blob_area  = 0
        for k in keypoints[i]:
            total_diameter = total_diameter + k.size
            total_blob_area = total_blob_area + 3.14 * math.pow(k.size/2, 2)
        if len(keypoints[i]):
            print ('avg diameter', total_diameter / len(keypoints[i]))
        else:
            print ('ERROR: avg diameter', 0)
        print ('aggregate blob area', total_blob_area)

        if i > 0:
            overlap = 0
            for k in keypoints[i]:
                for p in keypoints[0]:
                    if pdist(k.pt, p.pt) < (k.size + p.size) * 1.0 / 2.0:
                        overlap = overlap + 1.0
                        break
            if len(keypoints[i]):
                print ('overlap ratio', overlap / len(keypoints[i]))
            else:
                print ('ERROR: overlap ratio', 0)
            
        im_with_keypoints = cv2.drawKeypoints(cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB), keypoints[i], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(im_with_keypoints)
        plt.axis('off')
        plt.savefig("10_123.pdf", dpi=600,format='pdf')
        #plt.savefig('/media/sf_shared/blobed/test.pdf', dpi = 600, format='pdf')

        #plt.savefig('xgca_256_after_decimation_blobed.pdf', format='pdf')
        #plt.savefig('xgca_256_stats_preserving_blobed.pdf', format='pdf')
        #plt.show()





    #for kp in keypoints:
    #    print '(x, y)', int(kp.pt[0]), int(kp.pt[1])
    #    print 'diameter', int(kp.size)
    #    print 'strength', kp.response

    #print 'keypoint type', type(keypoints[0])
    # Show keypoints

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
    print len(r), len(z), len(conn), len(data)
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=25));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig(filename, format='png')

def psnr_c(original_data, base, leveldata_len, deci_ratio, level_id):
    for i in range(len(leveldata_len)-level_id,len(leveldata_len)):
        leveldata=np.zeros(leveldata_len[i])
        for j in range(leveldata_len[i]):
            index1=j//deci_ratio
            index2=j%deci_ratio
            if index1!= len(base)-1:
                if index2 != 0:
                    leveldata[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
            else:
                if index2 != 0:
                    leveldata[j]=(base[index1]*2)*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
        base=leveldata
    if len(base) !=len(original_data):
        print "len(leveldata) !=len(original_data)"

    MSE = 0.0
    for i in range(len(original_data)):
        #print i, original_data[i]-base[i]
        MSE=(original_data[i]-base[i])*(original_data[i]-base[i])+MSE
    MSE=MSE/len(original_data)
    if MSE < 1e-6:
        MSE=0.0
    if MSE ==0.0:
        print "Couldn't get PSNR of two identical data."
        return 0
    else:
        psnr=10*math.log(np.max(original_data)**2/MSE,10)
    #print "psnr=",psnr
    return psnr

def partial_refinement_new(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
    base_index = range(len(base))
    refined_base_index=[]
    print "len(base_index)=",len(base_index)
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    full_delta=range(finer_len)
    non_refine = list(set(full_delta).difference(set(chosn_index)))
    print "len(chosn_index)=",len(chosn_index)
    for i in range(len(chosn_index)):
        #if i%10000 ==0:
        #    print i
        index1 = chosn_index[i] // deci_ratio
        index2 = chosn_index[i] % deci_ratio
        if index1 != len(base)-1:
            if index2!=0:
                finer[chosn_index[i]]=(base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i]
                finer_p.append((base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i])
                finer_r_p.append((base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i])
                finer_z_p.append((base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i])
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print finer[chosn_index[i]], data[chosn_index[i]] 
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
            else:
                finer[chosn_index[i]] = base[index1]
                finer_r[chosn_index[i]] = base_r[index1]
                finer_z[chosn_index[i]] = base_z[index1]
                finer_p.append(base[index1])
                finer_r_p.append(base_r[index1])
                finer_z_p.append(base_z[index1])
                refined_base_index.append(index1)
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR2: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
        else:
            if index2!=0:
                finer[chosn_index[i]]= 2* base[index1]*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=2 * base_r[index1]*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=2 * base_z[index1]*index2/deci_ratio + chosn_z[i]
                finer_p.append(2* base[index1]*index2/deci_ratio + chosn_data[i])
                finer_r_p.append(2 * base_r[index1]*index2/deci_ratio + chosn_r[i])
                finer_z_p.append(2 * base_z[index1]*index2/deci_ratio + chosn_z[i])
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR3: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
            else:
                finer[chosn_index[i]] = base[index1]
                finer_r[chosn_index[i]] = base_r[index1]
                finer_z[chosn_index[i]] = base_z[index1]
                finer_p.append(base[index1])
                finer_r_p.append(base_r[index1])
                finer_z_p.append(base_z[index1])
                refined_base_index.append(index1)
                #base_index.remove(index1)
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR4: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
    for i in range(len(non_refine)):
        index1 = non_refine[i] // deci_ratio
        index2 = non_refine[i] % deci_ratio
        if index1 != len(base)-1:
            if index2!=0:
                finer[non_refine[i]]=(base[index1]+base[index1+1])*index2/deci_ratio
                finer_r[non_refine[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                finer_z[non_refine[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                #if (data[non_refine[i]] - finer[non_refine[i]] - delta[non_refine[i]])>0.000001:
                #    print "ERROR11:",data[non_refine[i]], finer[non_refine[i]], delta[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]] - delta_r[non_refine[i]])>0.000001:
                #    print "ERROR11:",r[non_refine[i]], finer_r[non_refine[i]], delta_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]] - delta_z[non_refine[i]])>0.000001:
                #    print "ERROR11:",z[non_refine[i]], finer_z[non_refine[i]], delta_z[non_refine[i]]

            else:
                finer[non_refine[i]] = base[index1]
                finer_r[non_refine[i]] = base_r[index1]
                finer_z[non_refine[i]] = base_z[index1]
                #if (data[non_refine[i]] - finer[non_refine[i]])>0.00001:
                #    print "ERROR22:",data[non_refine[i]], finer[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]])>0.00001:
                #    print "ERROR22:",r[non_refine[i]], finer_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]])>0.00001:
                #    print "ERROR22:",z[non_refine[i]], finer_z[non_refine[i]]
        else:
            if index2!=0:
                finer[non_refine[i]]= 2* base[index1]*index2/deci_ratio
                finer_r[non_refine[i]]=2* base_r[index1]*index2/deci_ratio
                finer_z[non_refine[i]]=2* base_z[index1]*index2/deci_ratio
                #if (data[non_refine[i]] - finer[non_refine[i]] - delta[non_refine[i]])>0.000001:
                #    print "ERROR33:",data[non_refine[i]], finer[non_refine[i]], delta[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]] - delta_r[non_refine[i]])>0.000001:
                #    print "ERROR33:",r[non_refine[i]], finer_r[non_refine[i]], delta_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]] - delta_z[non_refine[i]])>0.000001:
                #    print "ERROR33:",z[non_refine[i]], finer_z[non_refine[i]], delta_z[non_refine[i]]

            else:
                finer[non_refine[i]] = base[index1]
                finer_r[non_refine[i]] = base_r[index1]
                finer_z[non_refine[i]] = base_z[index1]
                #if (data[non_refine[i]] - finer[non_refine[i]])>0.00001:
                #    print "ERROR22:",data[non_refine[i]], finer[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]])>0.00001:
                #    print "ERROR22:",r[non_refine[i]], finer_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]])>0.00001:
                #    print "ERROR22:",z[non_refine[i]], finer_z[non_refine[i]]
    remain_base_index = list(set(base_index).difference(set(refined_base_index)))
    for i in remain_base_index:
        finer_p.append(base[i])
        finer_r_p.append(base_r[i])
        finer_z_p.append(base_z[i])
    return finer, finer_r, finer_z,finer_p, finer_r_p, finer_z_p

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
filename = "full_data.bin"
finer_len = 4992221
f = open(filename, "rb")
dpot_str = f.read(finer_len*8)
r_str = f.read(finer_len*8)
z_str = f.read(finer_len*8)
f.close()
number_of_original_elements = str(finer_len)
dpot=struct.unpack(number_of_original_elements+'d',dpot_str)
r=struct.unpack(number_of_original_elements+'d',r_str)
z=struct.unpack(number_of_original_elements+'d',z_str)

reduced_len = 4876
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

plot(dpot_L1,r_L1,z_L1,"dpot_L1.png") 
blob_detection("dpot_L1.png")

#fp = np.load("xgc/xgc_60_120_75_1.npz")
#data = fp['finer']
#r = fp['finer_r']
#z = fp['finer_z']    
#plot(data,r,z,"12345.png") 
deci_ratio = 1024
number_blocks = 25
block_interval = 1/number_blocks
block_num = 1
print "block_num=",block_num
finer_len = int(ioctx_2.stat("delta_L0_L1_o")[0]/8)
delta_index_L0_L1_str = ""
delta_L0_L1_str = ""
delta_r_L0_L1_str = ""
delta_z_L0_L1_str = ""
delta_len = 0
for i in range(int(block_num)):
    #print "block tag=",i*block_interval
    #print i*0.1
    chosn_delta_name= "delta_L0_L1_"+str(i*block_interval)
    chosn_delta_r_name = "delta_r_L0_L1_"+str(i*block_interval)
    chosn_delta_z_name = "delta_z_L0_L1_"+str(i*block_interval)
    chosn_index_name = "delta_index_" + str(i*block_interval)
    #print chosn_delta_name, chosn_delta_r_name, chosn_delta_z_name, chosn_index_name
    #if block_num != 1:
    delta_index_L0_L1_str += ioctx_2.read(chosn_index_name,ioctx_2.stat(chosn_index_name)[0],0)
    delta_L0_L1_str += ioctx_2.read(chosn_delta_name,ioctx_2.stat(chosn_delta_name)[0],0)
    delta_r_L0_L1_str += ioctx_2.read(chosn_delta_r_name,ioctx_2.stat(chosn_delta_r_name)[0],0)
    delta_z_L0_L1_str += ioctx_2.read(chosn_delta_z_name,ioctx_2.stat(chosn_delta_z_name)[0],0)
    delta_len += int(ioctx_2.stat(chosn_delta_name)[0]/8)
    #else:
    #   delta_index_L0_L1_str += ioctx_2.read(chosn_index_name,4,0)
    #   delta_L0_L1_str += ioctx_2.read(chosn_delta_name,8,0)
    #   delta_r_L0_L1_str += ioctx_2.read(chosn_delta_r_name,8,0)
    #   delta_z_L0_L1_str += ioctx_2.read(chosn_delta_z_name,8,0)
    #   delta_len += 1            
    #print ioctx_2.stat(chosn_index_name)[0]/4, ioctx_2.stat(chosn_delta_name)[0]/8, ioctx_2.stat(chosn_delta_r_name)[0]/8, ioctx_2.stat(chosn_delta_z_name)[0]/8
    #delta_len += int(ioctx_2.stat(chosn_delta_name)[0]/8) 

delta_index = struct.unpack(str(delta_len)+'i',delta_index_L0_L1_str)
delta_L0_L1 = struct.unpack(str(delta_len)+'d',delta_L0_L1_str)
delta_r_L0_L1 = struct.unpack(str(delta_len)+'d',delta_r_L0_L1_str)
delta_z_L0_L1 = struct.unpack(str(delta_len)+'d',delta_z_L0_L1_str)

finer, finer_r,finer_z, finer_p, finer_r_p,finer_z_p = partial_refinement_new(delta_index, finer_len, delta_L0_L1, delta_r_L0_L1, delta_z_L0_L1, dpot_L1, r_L1, z_L1, deci_ratio)
plot(finer_p,finer_r_p,finer_z_p,str(int(block_num))+"_123.png")

blob_detection(str(int(block_num))+"_123.png")
ioctx_2.close()
cluster.shutdown()











