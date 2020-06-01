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
def find_large_elements(source,threshold):
    max_gradient=0.0
    for i in range(len(source)):
        if math.fabs(source[i])> max_gradient:
            max_gradient = math.fabs(source[i])
     
    #s_mean = np.mean(source)
    #s_std = np.std(source,ddof=1)
    #print "source=",source
    #fig,ax = plt.subplots(figsize=(11,6))
    #if np.fabs(np.max(source)-s_mean) < np.fabs(np.min(source)-s_mean):
    #    interval = np.fabs(np.min(source)-s_mean)
    #else:
    #    interval = np.fabs(np.max(source)-s_mean)
    #plt.plot(source)
    #plt.hlines(s_mean + interval, 0, 2500000, 'r')
    #plt.hlines(s_mean - interval, 0, 2500000, 'b')
    #plt.savefig("source.pdf",format='pdf')
    
    #print threshold
    #high = s_mean + interval * threshold
    #low = s_mean - interval * threshold
    return max_gradient * threshold 

def find_augment_points_gradient(base,chosn_index,threshold):
    if threshold == 0.0:
        return [range(len(base))]
    elif threshold == 1.0:
        return []
    chosn_points=[]

    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    #print "-1\n"
    #print sys.getsizeof(base)/1024/1024
    base_gradient=np.gradient(base)
    #print "0\n"
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(base_gradient[j])
    #print "1\n"   
    thre = find_large_elements(chosn_points, threshold)
    #print "2\n"  
    #print high_b, low_b
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if math.fabs(base_gradient[j])> thre:
                #print "VIP=",j
                temp_1.append(j)
        #if len(temp_1)>1:
        temp_index.append(temp_1)
        temp_1=[]
    #print "temp_index=",temp_index                                          
    for i in temp_index:
        if len(i)>1:
            for j in range(1,len(i)):
                if i[j]-i[j-1]>1:
                    temp_interval.append(i[j]-i[j-1]) 
    #print "temp_interval=",temp_interval
    #print "temp_interval",temp_interval
    if len(temp_interval) ==1:
        max_intv = temp_interval[0]
    else: 
        max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    #print "max_intv=",max_intv
    #print temp_index                  
    temp_2=[]

    for i in temp_index:
        temp_2.append(i[0])
        if len(i) > 1:
            for j in range(1,len(i)):
                if i[j]-i[j-1] <= max_intv:
                    temp_2.append(i[j])
                else:
                    #if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
                    temp_2=[]
                    temp_2.append(i[j])
            #if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
            temp_2=[]
    #Not Finish
    last_tag = chosn_index_finer[0][-1]
    for i in chosn_index_finer:
        #before_len = len(i)
        tm = i[-1]
        if i[-1]!=len(dpot_L1)-1 and i[0]-last_tag>1:
            i.append(i[-1]+1)
        if i[0]!=0:
            i.append(i[0]-1)
        i.sort()
        last_tag = tm
    return chosn_index_finer

def partial_refinement(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    start = 0
    inc = 0
    i=0
    num_refine=0
    refine_index=[]
    while(i<len(chosn_index)):
        
        for m in range(start,chosn_index[i]):
            index1 = m // deci_ratio
            index2 = m % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    finer[m]=(base[index1]+base[index1+1])*index2/deci_ratio
                    finer_r[m]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                    finer_z[m]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[m]=2*base[index1]*index2/deci_ratio
                    finer_r[m]=2*base_r[index1]*index2/deci_ratio
                    finer_z[m]=2*base_z[index1]*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
        for n in range(chosn_index[i],chosn_index[i+1]+1):
            index1 = n // deci_ratio
            index2 = n % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            inc+=1
        start = chosn_index[i+1]+1
        if i == len(chosn_index)-2:
            for j in range(start,finer_len):
                index1 = j // deci_ratio
                index2 = j % deci_ratio
                if index1!=len(base)-1:
                    if index2!=0:
                        finer[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                        finer_r[j]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                        finer_z[j]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]
                        #finer_p.append(base[index1])
                        #finer_r_p.append(base_r[index1])
                        #finer_z_p.append(base_z[index1])
                else:
                    if index2!=0:
                        finer[j]=2*base[index1]*index2/deci_ratio
                        finer_r[j]=2*base_r[index1]*index2/deci_ratio
                        finer_z[j]=2*base_z[index1]*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
        i+=2
    #print refine_index    
    #print num_refine
    #print "percentage of selected points= ", num_refine/(finer_len-len(base))

    #for i in chosn_index:
    #    if i[-1] != len(base)-1:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,(i[-1]+1)*deci_ratio))
    #    else:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,len(delta_L1)))
    #PSNR=psnr(delta_L1,chosn_index_n,np.max(finer),len(delta_L1))
    #print finer
    #print 3
    return finer, finer_r, finer_z

def k_means(source,savefig,k):
    if k == "true":
        find_k(source,savefig)
    #print source
    y=np.array(source).reshape(-1,1)
        
    km=KMeans(n_clusters=2)
    km.fit(y)
    km_label=km.labels_
    #print "source=",source
    #print "km.label=",km_label
    #print km.cluster_centers_
    if len(km_label)!=len(source):
        print "length issue"
    sorted_cluster_index=np.argsort(km.cluster_centers_.reshape(-1,))
    #print "sorted_cluster_index=",sorted_cluster_index
    group=sorted_cluster_index[:1]
    #print "group=",group
    #group_index=[]

    #for i in range(len(km_label)):
    #    if km_label[i] in group: group_index.append(i)
    limit = 0
    for i in range(len(source)):
        if km_label[i] in group:
            #print km_label[i]
            if source[i]>limit:
                #print source[i],limit
                limit = source[i]
    return limit

def noise_threshold(noise):
    peak_noise = 0.1
    no_noise = 100
    k=(1.0-0.0)/(no_noise - peak_noise)
    b =1.0 - k * no_noise

    if noise < peak_noise:
        threshold = 0.0
    elif noise > no_noise:
        threshold = 1.0 
    else:
        threshold = noise*k+b
        #thre.append(threshold)

    return threshold

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def prediction_noise_wave(samples, hi_freq_ratio,timestep_interval):

    sample_rate = 1/timestep_interval
    Nsamples = len(samples)
    print "sample rate = ",sample_rate
    amp = fft.fft(samples)/(Nsamples/2.0)
    amp_complex_h = amp[range(int(len(samples)/2))]
    amp_h = np.absolute(amp_complex_h)
    
    freq=fft.fftfreq(amp.size,1/sample_rate)
    freq_h = freq[range(int(len(samples)/2))] 
    
    if amp_h[0]>1e-10:
    	threshold = np.max(np.delete(amp_h,0,axis=0))*hi_freq_ratio
    	dc = amp_h[0]/2.0
    	start_index = 1
    else:
    	threshold = np.max(amp_h)*hi_freq_ratio
    	dc = 0.0
    	start_index = 0
    #print "dc",dc
    #print "threshold",threshold
    selected_freq = []
    selected_amp = []
    selected_complex=[]
    for i in range(start_index,len(amp_h)):
    	if amp_h[i]>=threshold:
    		selected_freq.append(freq_h[i])
    		selected_amp.append(amp_h[i])
    		selected_complex.append(amp_complex_h[i])  
    
    selected_phase = np.arctan2(np.array(selected_complex).imag,np.array(selected_complex).real)
    
    for i in range(len(selected_phase)):
    	if np.fabs(selected_phase[i])<1e-10:
    		selected_phase[i]=0.0
    #print "future_timestep", future_timestep
    #future_timestep=np.array([0])
    return dc, selected_amp, selected_freq, selected_phase
    
def get_prediction_threshold(dc, selected_amp, selected_freq, selected_phase, time):
    sig = dc
    for i in range(len(selected_freq)):
        sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*time+ selected_phase[i])
    if sig < 0 :
        sig = 1.1
    print "Noise amplitude=",sig
    threshold = noise_threshold(sig)
    return threshold

def get_chosn_data_index(threshold):
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    #r_L1_compressed=f.read(2975952*8)
    #r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    #z_L1_compressed=f.read(2516984*8)
    #z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
     
    if threshold == 0.0:
        chosn_index_L1 = []            
    elif threshold == 1.0:
        chosn_index_L1 = [range(len(dpot_L1))]
    else:
        chosn_index_L1 = find_augment_points_gradient(dpot_L1,[range(len(dpot_L1))], 1-threshold)

    return chosn_index_L1    

def calc_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1,p2,p3
    #return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)
    area = abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))*0.5)
    return area

def high_potential_area(dpot,R,Z,thre):
    start = time.time()
    points = np.transpose(np.array([Z,R]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    area=0.0
    for i in range(len(conn)):
        index1=conn[i][0]
        index2=conn[i][1]
        index3=conn[i][2]
        #if (dpot[index1]>thre and dpot[index2]>thre) or (dpot[index1]>thre and dpot[index3]>thre) or (dpot[index2]>thre and dpot[index3]>thre):
        if (dpot[index1]+dpot[index2]+dpot[index3])/3.0 > thre:
            each_area=calc_area((R[index1],Z[index1]),(R[index2],Z[index2]),(R[index3],Z[index3]))
            area = area + each_area
    end = time.time()
    print "High potential analysis time = ", end - start
    return area

def fully_refine(deci_ratio,timestep,tag,update_interval,thre):
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
    #   print "Connected to the cluster."

    #if not cluster.pool_exists('tier0_pool'):
	#    raise RuntimeError('No data pool exists')
    #ioctx_0 = cluster.open_ioctx('tier0_pool')
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    r_L1_compressed=f.read(2975952*8)
    r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    z_L1_compressed=f.read(2516984*8)
    z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
    #dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    #r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    #z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
    #dpot_L1=zfpy._decompress(dpot_L1_str, 4, [10556545], tolerance=0.01)
    #r_L1=zfpy._decompress(r_L1_str, 4, [10556545], tolerance=0.01)
    #z_L1=zfpy._decompress(z_L1_str, 4, [10556545], tolerance=0.01)
    #tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
    #dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
    #r_L1=struct.unpack(tag+'d',r_L1_str)
    #z_L1=struct.unpack(tag+'d',z_L1_str)
    #ioctx_0.close()

    if not cluster.pool_exists('tier2_pool'):
	    raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    #delta_L0_L1_o_str = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
    #delta_L0_L1_o = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1_o")[0]/8))+'d',delta_L0_L1_o_str)
    start=time.time()
    delta_L0_L1_str = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
    #print "delta_L0_L1 size =", sys.getsizeof(delta_L0_L1)/1024/1024
    delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",ioctx_2.stat("delta_r_L0_L1_o")[0],0)
    delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",ioctx_2.stat("delta_z_L0_L1_o")[0],0)
    end = time.time()
    delta_read_time = end-start
    print "Delta reading time = ", delta_read_time
    #delta_L0_L1 = zfpy._decompress(delta_L0_L1_str, 4, [4992221], tolerance=0.01)
    #delta_r_L0_L1 = zfpy._decompress(delta_r_L0_L1_str, 4, [4992221], tolerance=0.01)
    #delta_z_L0_L1 = zfpy._decompress(delta_z_L0_L1_str, 4, [4992221], tolerance=0.01)
    delta_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1_o")[0]/8))+'d',delta_L0_L1_str)
    delta_r_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_r_L0_L1_o")[0]/8))+'d',delta_r_L0_L1_str)
    delta_z_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_z_L0_L1_o")[0]/8))+'d',delta_z_L0_L1_str)
    sample_bandwidth = (ioctx_2.stat("delta_L0_L1_o")[0]+ ioctx_2.stat("delta_r_L0_L1_o")[0] + ioctx_2.stat("delta_z_L0_L1_o")[0])/1024/1024/delta_read_time
    #print "Delta reading bandwidth = ", (ioctx_2.stat("delta_L0_L1")[0]+ ioctx_2.stat("delta_r_L0_L1")[0] + ioctx_2.stat("delta_z_L0_L1")[0])/1024/1024/delta_time
    fname = "sample_"+str(tag)+".npz"
        
    if timestep != 0:
        fp = np.load(fname)
        sample_bd = fp['sample_bandwidth']
        sample_read_time = fp['sample_read_time']
    else:
        sample_bd = np.array([]) 
        sample_read_time = np.array([])
    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()
    
    sample_bd.append(sample_bandwidth)
    sample_read_time.append(delta_read_time)
    if len(sample_bd) == update_interval:
        fname1 = "sample_"+str(tag+1)+".npz"
        print "Written to ", fname1
        np.savez(fname1, sample_bandwidth = np.array([sample_bandwidth]),sample_read_time = np.array([delta_read_time]))
    bdstr=''
    for i in range(len(sample_bd)-1):
        bdstr += str(sample_bd[i])+","
    bdstr += str(sample_bd[-1])
    print "Total bandwidth samples=",bdstr
    timestr=''
    for i in range(len(sample_read_time)-1):
        timestr += str(sample_read_time[i])+","
    timestr += str(sample_read_time[-1])
    print "Total time samples=",timestr
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))
    #tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
    #delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
    #print "delta_L0_L1 size =", sys.getsizeof(delta_L0_L1)/1024/1024

    #delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
    #delta_z_L0_L1 = struct.unpack(tag+'d',delta_z_L0_L1_str)
    #aa=time.time()
    #w_finer, w_finer_r,w_finer_z = whole_refine(dpot_L1,r_L1,z_L1,delta_L0_L1,delta_r_L0_L1,delta_z_L0_L1,deci_ratio)
    #bb=time.time()
    finer, finer_r,finer_z = partial_refinement((0,len(delta_L0_L1)-1), len(delta_L0_L1), delta_L0_L1, delta_r_L0_L1, delta_z_L0_L1, dpot_L1, r_L1, z_L1, deci_ratio)
    ioctx_2.close()
    cluster.shutdown()
    #print "start plot\n"
    #a=time.time()
    #b=time.time()
    #print "plot time = ", b-a
    #high_p_area = high_potential_area(finer, finer_r,finer_z,thre)
    #return high_p_area

def partial_refine(deci_ratio,timestep,psnr,ctag,update_interval,thre):
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

    #dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    #r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    #z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
    #dpot_L1=zfpy._decompress(dpot_L1_str, 4, [2496111], tolerance=0.01)
    #r_L1=zfpy._decompress(r_L1_str, 4, [2496111], tolerance=0.01)
    #z_L1=zfpy._decompress(z_L1_str, 4, [2496111], tolerance=0.01)
    #tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
    #dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
    #r_L1=struct.unpack(tag+'d',r_L1_str)
    #z_L1=struct.unpack(tag+'d',z_L1_str)
    #sa = time.time()

    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    r_L1_compressed=f.read(2975952*8)
    r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    z_L1_compressed=f.read(2516984*8)
    z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()

    fname = "recontructed_noise_"+str(ctag-1)+".npz"
    f_np = np.load(fname)
    dc = f_np["recontructed_noise_dc"][0]
    s_freq = f_np["recontructed_noise_freq"]
    s_amp = f_np["recontructed_noise_amp"]
    s_phase = f_np["recontructed_noise_phase"]
    thre = get_prediction_threshold(dc, s_amp, s_freq, s_phase, timestep)
    print "threshold=",thre
    chosn_index = get_chosn_data_index(thre)
    
    chosn_index_e = []
    for i in range(len(chosn_index)):
        chosn_index_e.append(chosn_index[i][0]*deci_ratio)
        chosn_index_e.append(chosn_index[i][-1]*deci_ratio)
    chosn_data_str = ""
    chosn_r_str = ""
    chosn_z_str = ""
    chosn_len = 0
    delta_read_time = 0.0
    i=0
    segment_num = 0
    segment_size = []
    #print "chosn_index_e",chosn_index_e
    while(i<len(chosn_index_e)):
        start = time.time()
        temp_str = ioctx_2.read("delta_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8)
        temp_r_str = ioctx_2.read("delta_r_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8)
        temp_z_str = ioctx_2.read("delta_z_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8) 
        end = time.time()
        delta_read_time += end-start
        #check overflow
        chosn_data_str = chosn_data_str + temp_str
        chosn_r_str =  chosn_r_str + temp_r_str
        chosn_z_str =  chosn_z_str + temp_z_str 
        chosn_len+= chosn_index_e[i+1]-chosn_index_e[i]+1
        segment_size.append((chosn_index_e[i+1]-chosn_index_e[i]+1)*8+(chosn_index_e[i+1]-chosn_index_e[i]+1)*8+(chosn_index_e[i+1]-chosn_index_e[i]+1)*8)
        segment_num += 1	
        i+=2
    chosn_data = struct.unpack(str(chosn_len)+'d',chosn_data_str)
    chosn_r = struct.unpack(str(chosn_len)+'d',chosn_r_str)
    chosn_z = struct.unpack(str(chosn_len)+'d',chosn_z_str)

    print "read time=",delta_read_time
    print "Number of segment = ",segment_num
    print "All segment size = ",segment_size
    print "Number of selected element=",(len(chosn_data) + len(chosn_r) + len(chosn_z))
    sample_bandwidth = (len(chosn_data) + len(chosn_r) + len(chosn_z))*8/1024/1024/delta_read_time
     
    finer_len = int(ioctx_2.stat("delta_L0_L1_o")[0]/8)
    ioctx_2.close()
    cluster.shutdown()
    print "Delta fetching percentage=",len(chosn_data)*100/finer_len
    fname = "sample_"+str(ctag)+".npz"
    if timestep == 0:
    	print "Error, time step for partial refinement couldn't be 0!\n"
    
    fp = np.load(fname)
    sample_bd = fp['sample_bandwidth']
    sample_read_time = fp['sample_read_time']
    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()

    sample_bd.append(sample_bandwidth)
    sample_read_time.append(delta_read_time)
    if len(sample_bd) == update_interval:
        fname1 = "sample_"+str(ctag+1)+".npz"
        print "Written to ", fname1
        np.savez(fname1, sample_bandwidth = np.array([sample_bandwidth]),sample_read_time = np.array([delta_read_time]))
    bdstr=''
    for i in range(len(sample_bd)-1):
        bdstr += str(sample_bd[i])+","
    bdstr += str(sample_bd[-1])
    print "Total bandwidth samples=",bdstr
    timestr=''
    for i in range(len(sample_read_time)-1):
        timestr += str(sample_read_time[i])+","
    timestr += str(sample_read_time[-1])
    print "Total time samples=",timestr
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))

    #aa=time.time() 
    finer, finer_r,finer_z = partial_refinement(chosn_index_e, finer_len, chosn_data,chosn_r,chosn_z,dpot_L1,r_L1,z_L1,deci_ratio)
    #bb =time.time()
    #print "Time for one time partial refinement=",bb-aa
    #high_p_area = high_potential_area(finer, finer_r,finer_z,thre)
    
    if psnr =="True":
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

        dpot_str = ioctx_2.read("data",ioctx_2.stat("data")[0],0)
        r_str = ioctx_2.read("r",ioctx_2.stat("r")[0],0)
        z_str = ioctx_2.read("z",ioctx_2.stat("z")[0],0)
        dpot=zfpy._decompress(dpot_str, 4, [21113089], tolerance=0.01)
    	r=zfpy._decompress(r_str, 4, [21113089], tolerance=0.01)
    	z=zfpy._decompress(z_str, 4, [21113089], tolerance=0.01)
        #tag=str(int(ioctx_0.stat("data")[0]/8))
        #dpot=struct.unpack(tag+'d',dpot_str)
        #r=struct.unpack(tag+'d',r_str)
        #z=struct.unpack(tag+'d',z_str)
        data_len=[len(dpot_L1),len(dpot)]
        #plot(dpot, r, z, "original")
        #print len(finer_L0)
        #print len(dpot)
        #np.append(finer_L0,finer_L0[-1])
        #cnt = 0
        #print r_L1[366], r_L1[367],r_L1[368]
        #print chosn_r_L0[367],chosn_r_L0[368]

        #for i in range(len(chosn_r_L0)):
        #    if finer_r_L0[i] -  r[i] > 1e-10:
        #        if i == 733:
        #            print i,finer_r_L0[i], r[i],"hello"
                        #print "wrong" 
        #        cnt+=1
        #print cnt
        ioctx_2.close()      
        psnr_finer=psnr_c(dpot, finer, data_len, deci_ratio, 1)
        psnr_original=psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
        print "finer PSNR=",psnr_finer
        print "original PSNR=",psnr_original
    #return high_p_area
 
def work_flow(deci_ratio, time_interval, psnr, update_interval,high_potential_threshold):
    tag = 0
    timestep = 0
    last_tag = 0
    for i in range(0,10000,time_interval):
        start=time.time()
        print "time = %d\n" %(i)
        if int(i/time_interval) %(update_interval-1) == 1 and i !=time_interval:
            print "start updating prediction"
            a=time.time()
            fname="sample_"+str(tag)+".npz"
            fp = np.load(fname)
            samples_bd = fp['sample_bandwidth']
            
            dc, s_amp, s_freq, s_phase = prediction_noise_wave(samples_bd, 0.5, time_interval)
            r_noise_name = "recontructed_noise_"+str(tag)+".npz"
            np.savez(r_noise_name, recontructed_noise_dc = np.asarray([dc]), recontructed_noise_amp = np.asarray(s_amp),recontructed_noise_freq = np.asarray(s_freq), recontructed_noise_phase = np.asarray(s_phase))
            tag += 1
            b=time.time()
            print "Updating prediction time = ", b-a
        if tag == 0:
            print "fully refinement\n"
            print i
            fully_refine(deci_ratio,i, tag, update_interval,high_potential_threshold)
        else:
            timestep = i % ((update_interval-1)*time_interval) 
            if timestep ==0:
				timestep = (update_interval-1)*time_interval 
            print "partial refinement\n"
            print timestep
            partial_refine(deci_ratio, timestep, psnr, tag,update_interval,high_potential_threshold)         
        end = time.time()
         
        #timestep += time_interval
        print "Analysis time = ",end-start
        time.sleep(time_interval-(end-start)) 
        #e=time.time()
        #print "One time workflow start from %d = %f\n"%(i*time_interval,e-a)

deci_ratio=2
Nsamples = 256
time_interval = 25
frequency_cut_off = 0.5
sampling_interval = 40000
q = Queue.LifoQueue()

work_flow(deci_ratio,time_interval,"false",49,40)
#partial_refine(deci_ratio,300,"false",2,49,40)
