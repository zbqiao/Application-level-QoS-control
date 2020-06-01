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
    s_mean = np.mean(source)
    s_std = np.std(source,ddof=1)
    #print "source=",source
    fig,ax = plt.subplots(figsize=(11,6))
    if np.fabs(np.max(source)-s_mean) > np.fabs(np.min(source)-s_mean):
        interval = np.fabs(np.min(source)-s_mean)
    else:
        interval = np.fabs(np.max(source)-s_mean)
    #plt.plot(source)
    #plt.hlines(s_mean + interval, 0, 2500000, 'r')
    #plt.hlines(s_mean - interval, 0, 2500000, 'b')
    #plt.savefig("source.pdf",format='pdf')
    
    print threshold
    high = s_mean + 0.6*interval * threshold
    low = s_mean - 0.6*interval * threshold
    return high,low

def find_augment_points_gradient(base,chosn_index,deci_ratio,threshold):
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
    high_b, low_b = find_large_elements(chosn_points, threshold)
    #print "2\n"  
    #print high_b, low_b
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if base_gradient[j]>=high_b or base_gradient[j]<low_b:
                #print "VIP=",j
                temp_1.append(j)
        if len(temp_1)>1:
            temp_index.append(temp_1)
        temp_1=[]
    #print "temp_index=",temp_index                                          
    for i in temp_index:
        for j in range(1,len(i)):
            if i[j]-i[j-1]>1:
                temp_interval.append(i[j]-i[j-1]) 
    #print "temp_interval=",temp_interval
    #print "temp_interval",temp_interval
    max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    #print "max_intv=",max_intv
    #print temp_index                  
    temp_2=[]

    for i in temp_index:
        temp_2.append(i[0])
        for j in range(1,len(i)):
            if i[j]-i[j-1] <= max_intv:
                temp_2.append(i[j])
            else:
                #print temp_2
                if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
                temp_2=[]
                temp_2.append(i[j])
        if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
        temp_2=[]
    #cnt=0
    #for i in range(len(chosn_index_finer)):
    #    for j in range(len(chosn_index_finer[i])):
    #        cnt+=1   
    #print "number of chosn index=",cnt
    #print np.shape(chosn_index_finer)
    return chosn_index_finer

def partial_refinement(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)

    #print finer_r[735]
    #finer_p=[]
    #finer_r_p=[]
    #finer_z_p=[]
    #print chosn_index
    start=0
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


def partial_refinementi_old(chosn_index, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros((len(base)-1)*deci_ratio+1)
    finer_r=np.zeros((len(base)-1)*deci_ratio+1)
    finer_z=np.zeros((len(base)-1)*deci_ratio+1)
    #finer_p=[]
    #finer_r_p=[]
    #finer_z_p=[]
    #print chosn_index
    start=0
    inc = 0
    i=0
    num_refine=0
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
                    #print i,k
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[inc]
                    num_refine+=1

                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chson_data[i][inc]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chson_r[i][inc]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chson_z[i][inc]   
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            inc+=1
        start = chosn_index[i+1]+1
        if i == len(chosn_index)-2:
            for j in range(start,(len(base)-1)*deci_ratio+1):
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
    print "percentage of selected points= ", num_refine/(len(delta_L1)-len(base))
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
    km=KMeans(n_clusters=5)
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
    group_index=[]

    for i in range(len(km_label)):
        if km_label[i] in group: group_index.append(i)
    limit = 0
    for i in range(len(source)):

        if km_label[i] in group:
            #print km_label[i]
            if source[i]>limit:
                #print source[i],limit
                limit = source[i]
    return limit

def noise_threshold(noise):
    peak_noise = 2
    no_noise = 0.5
    thre=[]
    k=(1.0-0.0)/(no_noise - peak_noise)
    b =1.0 - k * no_noise
    for i in range(len(noise)):
        if noise[i] < 0.0:
            noise[i]=no_noise
    	if noise[i] > peak_noise:
        	threshold = 0.0
    	elif noise[i] < no_noise:
        	threshold = 1.0 
    	else:
        	threshold = noise[i]*k+b
        thre.append(threshold)

    return thre

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def plot(data,r,z,filename):
    points = np.transpose(np.array([z,r]))
    #print points
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
	#plt.figure(206, figsize=(10,7))
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
    #plt.savefig(filename+'.png', format='png')

def whole_refine(base,base_r,base_z,delta_L1,delta_L1_r,delta_L1_z,deci_ratio):
    finer=np.zeros(len(delta_L1),dtype = np.float64)
    finer_r=np.zeros(len(delta_L1),dtype = np.float64)
    finer_z=np.zeros(len(delta_L1),dtype = np.float64)
    #print "start fully\n"
    #cnt = 0
    for i in range(len(delta_L1)):
        
        index1 = i//deci_ratio
        index2 = i%deci_ratio
        if index1!=len(base)-1:
            if index2!=0:
                finer[i]=(base[index1]+base[index1+1])*index2/deci_ratio
                finer_r[i]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                finer_z[i]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
            else:
                finer[i]=base[index1]
                finer_r[i]=base_r[index1]
                finer_z[i]=base_z[index1]
        else:
            if index2!=0:
                finer[i]=2*base[index1]*index2/deci_ratio
                finer_r[i]=2*base_r[index1]*index2/deci_ratio
                finer_z[i]=2*base_z[index1]*index2/deci_ratio
            else:
                finer[i]=base[index1]
                finer_r[i]=base_r[index1]
                finer_z[i]=base_z[index1]
        #cnt+=1
    #print "total loops of fully refine=",cnt
    #print "end fully\n"
    return finer,finer_r,finer_z

def sampling(small_block_name, Nsamples,write_out_tag):
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
    
    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx = cluster.open_ioctx('tier2_pool')
    
    prefix_time=""
    y=[]
    for i in range(Nsamples):
    #for i in range(Nsamples):
        a =time.time()

        test_sample_str = ioctx.read(small_block_name, ioctx.stat(small_block_name)[0],0)
	#tag = :str(int(ioctx_2.stat("test_sample")[0]/8))
	#delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
        e=time.time()
        #print e-a
        y.append(e-a)
    ioctx.close()
    cluster.shutdown()
    if write_out_tag =="true":
        for i in range(len(y)-1):
            prefix_time += str(y[i])+","
        prefix_time += str(y[-1])
        file=open("result_fully.txt","w")
        file.write(prefix_time)
        file.close()

    x = []
    x1 = 0
    for i in y:
       	x1+= i
       	x.append(x1) 
        #x1+= i/2
    print "End of the prediction period=",x[-1] 
    return x,y
def prediction_noise_wave(Nsamples, hi_freq_ratio, write_out_y_tag,timestep_interval):
    x,y = sampling("all_diff", Nsamples, write_out_y_tag)

    sample_rate = Nsamples/x[-1]
    print "sample rate = ",sample_rate
    amp = fft.fft(y)/(Nsamples/2.0)
    amp_complex_h = amp[range(int(len(x)/2))]
    amp_h = np.absolute(amp_complex_h)
    
    freq=fft.fftfreq(amp.size,1/sample_rate)
    freq_h = freq[range(int(len(x)/2))] 
    
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
    future_time = np.arange(0,int(x[-1]),timestep_interval)
    #print "future_timestep", future_timestep
    #future_timestep=np.array([0])
    return dc, selected_amp, selected_freq, selected_phase,future_time,x[-1]
    
def get_prediction_threshold(dc, selected_amp, selected_freq, selected_phase, time):
    sig = dc
    for i in range(len(selected_freq)):
        sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*time+ selected_phase[i])
    print "Noise amplitude=",sig
    threshold = noise_threshold(sig)
    return threshold

def write_chosn_data(threshold,timesteps,deci_ratio,ctag):
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    r_L1_compressed=f.read(2975952*8)
    r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    z_L1_compressed=f.read(2516984*8)
    z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close() 
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

    #dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    #dpot_L1=zfpy._decompress(dpot_L1_str, 4, [2496111], tolerance=0.01)
    #print sys.getsizeof(dpot_L1)/1024/1024
   

    #print dpot_L1
    #dpot_L1_o_str = ioctx_0.read("dpot_L1_o",ioctx_0.stat("dpot_L1_o")[0],0)
    #tag=str(int(ioctx_0.stat("dpot_L1_o")[0]/8))
    #dpot_L1_o=struct.unpack(tag+'d',dpot_L1_o_str)
    #print sys.getsizeof(dpot_L1_o)/1024/1024
    
    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    delta_L0_L1_str = ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0],0)
    delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
    delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0],0)
    delta_L0_L1 = zfpy._decompress(delta_L0_L1_str, 4, [4992221], tolerance=0.01)
    delta_r_L0_L1 = zfpy._decompress(delta_r_L0_L1_str, 4, [4992221], tolerance=0.01)
    delta_z_L0_L1 = zfpy._decompress(delta_z_L0_L1_str, 4, [4992221], tolerance=0.01)
    #delta_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1")[0]/8))+'d',delta_L0_L1_str)
    #delta_r_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_r_L0_L1")[0]/8))+'d',delta_r_L0_L1_str)
    #delta_z_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_z_L0_L1")[0]/8))+'d',delta_z_L0_L1_str)
    #for m in range(1):
    for m in range(len(threshold)):
        chosn_index_group=[]
        if threshold[m]  == 0.0:
            chosn_index_group=[]
        elif threshold[m] == 1.0:
            chosn_index_group=[0,len(delta_L0_L1)-1]
        else:
            chosn_index_L1 = find_augment_points_gradient(dpot_L1,[range(len(dpot_L1))],deci_ratio, 1-threshold[m])
            for i in range(len(chosn_index_L1)):
                chosn_index_group.append(chosn_index_L1[i][0] * deci_ratio)
                chosn_index_group.append(chosn_index_L1[i][-1] * deci_ratio)

        chosn_data_L0=[]
        chosn_r_L0=[]
        chosn_z_L0=[]
        if threshold[m] ==1.0:
            chosn_data_L0 = delta_L0_L1
            chosn_r_L0 = delta_r_L0_L1
            chosn_z_L0 = delta_z_L0_L1
        else:
            for i in chosn_index_L1:
                for j in range(i[0]*deci_ratio, i[-1]*deci_ratio+1):
                    chosn_data_L0.append(delta_L0_L1[j])
                    chosn_r_L0.append(delta_r_L0_L1[j])
                    chosn_z_L0.append(delta_z_L0_L1[j])

        #print "Points selection percentage =",len(chosn_data_L0)/len(delta_L0_L1)
        #print chosn_index_group_L0
        print "delta fetching percentage=",len(chosn_data_L0)*100/len(delta_L0_L1)
        
        chosn_index_group_name="chosn_index_group_"+ str(timesteps[m])+"_"+str(ctag)
        chosn_data_name = "chosn_data_"+ str(timesteps[m])+"_"+str(ctag)
        chosn_r_name = "chosn_r_" + str(timesteps[m])+"_"+str(ctag)
        chosn_z_name = "chosn_z_" + str(timesteps[m])+"_"+str(ctag)
        
        fname = str(timesteps[m])+"_"+str(ctag)+".npz"
        #f = open(fname,"wb")
        np.savez(fname, chosn_index_group = np.array(chosn_index_group), finer_len = np.array([len(delta_L0_L1)])) 
        #ioctx_0.write_full(chosn_index_group_name, struct.pack(str(len(chosn_index_group))+'i',*chosn_index_group))
        #ioctx_0.set_xattr(chosn_index_group_name, "finer_len", str(len(delta_L0_L1)))
        ioctx_2.write_full(chosn_data_name, struct.pack(str(len(chosn_data_L0))+'d',*chosn_data_L0))
        ioctx_2.write_full(chosn_r_name, struct.pack(str(len(chosn_r_L0))+'d',*chosn_r_L0))
        ioctx_2.write_full(chosn_z_name, struct.pack(str(len(chosn_z_L0))+'d',*chosn_z_L0))
    ioctx_2.close()
    cluster.shutdown()
def fetch_from_HDD(times):
    a=time.time()
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

    if not cluster.pool_exists('tier0_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_0 = cluster.open_ioctx('tier0_pool')
    
    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    y=[]
    print "ssd_hdd:hdd:",ioctx_2.stat("delta_L0_L1")[0]/1024/1024
    print "ssd_hdd:ssd",ioctx_0.stat("dpot_L1")[0]/1204/1024
    for i in range(times):
        cc=time.time()
    	dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    	#r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    	#z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
        #aa=time.time() 
        #print "ssd time =",aa-cc
    	delta_L0_L1_str = ioctx_2.read("delta_L0_L1_qua",ioctx_2.stat("delta_L0_L1_qua")[0],0)
    	#delta_r_L0_L1_str= ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0],0)
    	#delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0],0)
        bb=time.time()
        #if bb-aa >1.0:
       	#	print "noise\n"
        #print "hdd time=",bb-cc
        y.append(bb-cc)
    ioctx_0.close() 
    ioctx_2.close()
    cluster.shutdown()
    b=time.time()
    print "Runtime for fetching base data from SSD and deltas from HDD=",b-a
    prefix_time=""
    for i in range(len(y)-1):
        prefix_time += str(y[i])+","
    prefix_time += str(y[-1])
    file=open("result_fully.txt","w")
    file.write(prefix_time)
    file.close()

def fetch_from_HDD_n(times):
    a=time.time()
    for i in range(times):
        fetch_from_HDD() 
    b=time.time()
    print "Runtime for fetching base data from SSD and deltas from HDD=",b-a

def all_fetch_from_HDD(dataname,times):
    a=time.time()
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

    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    print "all hdd:",ioctx_2.stat(dataname)[0]/1024/1024 
    y=[]
    #prefix_time=''
    for i in range(times):
        aa=time.time()   
    	data_str = ioctx_2.read(dataname,ioctx_2.stat(dataname)[0],0)
        #tag = str(int(ioctx_2.stat(dataname)[0]/8))
    	#data = struct.unpack(tag+'d',data_str)
        #for i in range(int(ioctx_2.stat(dataname)[0]/8/2)):
        #for i in range(int(ioctx_2.stat(dataname)[0]/8/2),int(ioctx_2.stat(dataname)[0]/8)):
        #    print data[i]
    	#r_str= ioctx_2.read(rname,ioctx_2.stat(rname)[0],0)
        #r = struct.unpack(tag+'d',r_str)
    	#z_str = ioctx_2.read(zname,ioctx_2.stat(zname)[0],0)
        #z = struct.unpack(tag+'d',z_str)
        bb=time.time()
        #print bb-aa
        y.append(bb-aa)
   
    #for i in range(len(y)-1):
    #        prefix_time += str(y[i])+","
    #prefix_time += str(y[-1])
    #tag = str(int(ioctx_2.stat(dataname)[0]/8))
    #data = struct.unpack(tag+'d',data_str)
    #file=open("results/result_fully_%s.txt"%(dataname),"a")
    #file.write(prefix_time)
    #file.close()
    #tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
    #delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
    #print delta_L0_L1
    #print ioctx_2.stat("data")[0]
    ioctx_2.close()
    cluster.shutdown()
    b=time.time()
    print "Runtime for fetching all data from HDD=",b-a
    prefix_time=""
    for i in range(len(y)-1):
        prefix_time += str(y[i])+","
    prefix_time += str(y[-1])
    file=open("result_fully.txt","w")
    file.write(prefix_time)
    file.close()

def fully_refine(deci_ratio,blob_tag,timestep):
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
    delta_L0_L1_str = ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0],0)
    #print "delta_L0_L1 size =", sys.getsizeof(delta_L0_L1)/1024/1024
    delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
    delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0],0)
    end = time.time()
    delta_time = end-start
    #print "Delta reading time = ", delta_time
    delta_L0_L1 = zfpy._decompress(delta_L0_L1_str, 4, [4992221], tolerance=0.01)
    delta_r_L0_L1 = zfpy._decompress(delta_r_L0_L1_str, 4, [4992221], tolerance=0.01)
    delta_z_L0_L1 = zfpy._decompress(delta_z_L0_L1_str, 4, [4992221], tolerance=0.01)
    sample_bandwidth = (ioctx_2.stat("delta_L0_L1")[0]+ ioctx_2.stat("delta_r_L0_L1")[0] + ioctx_2.stat("delta_z_L0_L1")[0])/1024/1024/delta_time
    #print "Delta reading bandwidth = ", (ioctx_2.stat("delta_L0_L1")[0]+ ioctx_2.stat("delta_r_L0_L1")[0] + ioctx_2.stat("delta_z_L0_L1")[0])/1024/1024/delta_time
    fname = "sample.npz"
    if timestep // 200 == 0:
        fp = np.load(fname)
        samples_bd = fp['samples_bandwidth']
        
    if timestep != 0:
        fp = np.load(fname)
        samples_bd = fp['samples_bandwidth']
        samples_time = fp['samples_time']
    else:
        samples_bd = np.array([]) 
        samples_time = np.array([])
    samples_bd = samples_bd.tolist()
    samples_time = samples_time.tolist()
    
    samples_bd.append(sample_bandwidth)
    samples_time.append(delta_time) 
    bdstr=''
    for i in range(len(samples_bd)-1):
        bdstr += str(samples_bd[i])+","
    bdstr += str(samples_bd[-1])
    print "Total bandwidth samples=",bdstr
    timestr=''
    for i in range(len(samples_time)-1):
        timestr += str(samples_time[i])+","
    timestr += str(samples_time[-1])
    print "Total time samples=",timestr
    np.savez(fname, samples_bandwidth = np.array(samples_bd), samples_time = np.array(samples_time))
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
    if blob_tag == "true":
        print "start plot\n"
        a=time.time()
        fname="original"
        plot(finer, finer_r, finer_z, fname)
        blob_detection(fname)
        b=time.time()
        print "Blob detection and save plot time=",b-a
    #b=time.time()
    #print "plot time = ", b-a

def fully_refine_n (deci_ratio, times, blob_tag):
    start = time.time()
    for i in range(times):
        fully_refine(deci_ratio,blob_tag)
    end = time.time()
    print "Fully nrefinement time = ", end-start
    
def reduced_data():
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

    if not cluster.pool_exists('tier0_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_0 = cluster.open_ioctx('tier0_pool')

    dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
    dpot_L1=zfpy._decompress(dpot_L1_str, 4, [2496111], tolerance=0.01)
    r_L1=zfpy._decompress(r_L1_str, 4, [2496111], tolerance=0.01)
    z_L1=zfpy._decompress(z_L1_str, 4, [2496111], tolerance=0.01)

    ioctx_0.close()
    cluster.shutdown()
    plot(dpot_L1, r_L1, z_L1, "opencv-python-color-detection/0_finer")

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
    name_hat = "beforeblob/"
    name_tail1 = "original"
    #name_tail2 = "original"
    name_tail2 = "finer_0"
    #print (name_tail2)
    name1 = name_hat+name_tail1+".png"
    name2 = name_hat + fname + ".png"
    image.append(cv2.imread("beforeblob/original.png"))
    image.append(cv2.imread(name2))
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
    print(len(image))
    for i in range(len(image)):
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
            print ('avg diameter', 0)
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
                print ('overlap ratio', 0)
            
        im_with_keypoints = cv2.drawKeypoints(cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB), keypoints[i], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(im_with_keypoints)
        plt.axis('off')
        plt.savefig('blobed/'+fname+'_blobed.pdf',format='pdf')

def partial_refine(deci_ratio,timestep,psnr,ctag,blob_tag):
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

    #if not cluster.pool_exists('tier0_pool'):
    #    raise RuntimeError('No data pool exists')
    #ioctx_0 = cluster.open_ioctx('tier0_pool')

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

    fname = str(timestep)+"_"+str(ctag)+".npz"
    f_np = np.load(fname)
    chosn_index = f_np["chosn_index_group"]
    finer_len = f_np["finer_len"][0]
    #chosn_index_group_name = "chosn_index_group_" + str(timestep)+"_"+str(ctag)
    #chosn_index_group_str = ioctx_0.read(chosn_index_group_name, ioctx_0.stat(chosn_index_group_name)[0],0)
    #chosn_index = struct.unpack(str(int(ioctx_0.stat(chosn_index_group_name)[0]/4))+'i',chosn_index_group_str)
    #finer_len_str=ioctx_0.get_xattr(chosn_index_group_name, "finer_len")
    #finer_len = int(finer_len_str)
    #afa = time.time()
    ##print "extra time for partial=",afa-sa
    chosn_data_name = "chosn_data_" + str(timestep) +"_"+str(ctag)
    chosn_r_name = "chosn_r_" + str(timestep)+"_"+str(ctag)
    chosn_z_name = "chosn_z_" + str(timestep)+"_"+str(ctag)
    if chosn_index ==():
        finer = dpot_L1
        finer_r = r_L1
        finer_z = z_L1
    else:
        if not cluster.pool_exists('tier2_pool'):
            raise RuntimeError('No data pool exists')
        ioctx_2 = cluster.open_ioctx('tier2_pool')
        chosn_data_str = ioctx_2.read(chosn_data_name, ioctx_2.stat(chosn_data_name)[0],0)
        chosn_r_str = ioctx_2.read(chosn_r_name, ioctx_2.stat(chosn_r_name)[0],0)
        chosn_z_str = ioctx_2.read(chosn_z_name, ioctx_2.stat(chosn_z_name)[0],0)

        tag=str(int(ioctx_2.stat(chosn_data_name)[0]/8))
        chosn_data = struct.unpack(tag+'d',chosn_data_str)
        #print "chosn_data size =", sys.getsizeof(chosn_data)/1024/1024
        chosn_r = struct.unpack(tag+'d',chosn_r_str)
        chosn_z = struct.unpack(tag+'d',chosn_z_str)
        #delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
        #tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
        #delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
        ioctx_2.close()
        #aa=time.time() 
        finer, finer_r,finer_z = partial_refinement(chosn_index, finer_len, chosn_data,chosn_r,chosn_z,dpot_L1,r_L1,z_L1,deci_ratio)
        #bb =time.time()
        #print "Time for one time partial refinement=",bb-aa
    #ioctx_0.close()
    cluster.shutdown()
    #print len(finer)
    #plot(finer, finer_r, finer_z, "opencv-python-color-detection/25_finer")

    if blob_tag == "true":
        a=time.time()
        fname=str(timestep) +"_"+str(ctag)
        plot(finer, finer_r, finer_z, fname)
        blob_detection(fname)
        b=time.time()
        print "Blob detection and save plot time=",b-a

    
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
def plot_original():
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
    dpot=zfpy._decompress(dpot_str, 4, [4992221], tolerance=0.01)
    r=zfpy._decompress(r_str, 4, [4992221], tolerance=0.01)
    z=zfpy._decompress(z_str, 4, [4992221], tolerance=0.01)
    plot(dpot, r, z, "opencv-python-color-detection/original")
    ioctx_2.close()

def partial_refine_n(deci_ratio,timestep,psnr,ctag,times):
    start = time.time()
    for i in range(times):
        partial_refine(deci_ratio,timestep,psnr,ctag,blob_tag)
    end = time.time()
    print "Partial nrefinement time at timestep %d = %f\n "%(timestep, end-start)
def get_chosn_data(Nsamples, deci_ratio, time_interval, write_out_tag, frequency_cut_off, tag,q):
    start=time.time()  
    dc, s_amp, s_freq, s_phase,future_time,prediction_period = prediction_noise_wave(Nsamples, frequency_cut_off, write_out_tag, time_interval)
    threshold = get_prediction_threshold(dc, s_amp, s_freq, s_phase, future_time)
    print "future time=",future_time
    print "threshold=",threshold
    write_chosn_data(threshold, future_time, deci_ratio,tag)
    q.put(tag)
    end=time.time()
    return end-start, prediction_period
def update_sampling(sampling_interval,Nsamples, deci_ratio, time_interval, write_out_tag, frequency_cut_off, q):
    tag =1
    for i in range(1,10):
        #a=time.time()
        #start = time.time()
    	get_chosn_data_time, prediction_period = get_chosn_data(Nsamples, deci_ratio, time_interval, write_out_tag, frequency_cut_off,tag, q)
        #end = time.time()
        print "Finish run %d st time sampling, it takes %f s, prediction period = %f s\n"%(i,get_chosn_data_time,prediction_period)
        #print "End time of %d st sampling=%f\n"%(i,end)
        tag+=1
        time.sleep(sampling_interval-get_chosn_data_time)
        #e=time.time()
        #print "One time sampling = %f\n"%(e-a)
        #print "End time of %d st entire sampling=%f\n"%(i,e)
def work_flow(deci_ratio, time_interval, psnr, blob_tag,q):
    tag = 0
    timestep = 0
    last_tag = 0
    for i in range(1000):
        start=time.time()
        print "time = %d\n" %(i*time_interval)
        if q.qsize() >0:
            tag=q.get()
            q.put(tag)
        
        if tag == 0:
            print "fully refinement\n"
            #fully_refine(deci_ratio)
            fully_refine(deci_ratio,blob_tag,i*time_interval)
        else:
            if tag != last_tag:
                timestep = 0
            print "partial refinement, tag = %d\n"%tag
            #partial_refine(deci_ratio, timestep, psnr, tag)  
            partial_refine_n(deci_ratio, timestep, psnr, tag, 5, blob_tag)         
        end = time.time()
        last_tag=tag 
        timestep += time_interval
        print "Analysis time = ",end-start
        time.sleep(time_interval-(end-start)) 
        #e=time.time()
        #print "One time workflow start from %d = %f\n"%(i*time_interval,e-a)
        #print "End time of %d st entire refinement=%f\n"%(i+1,e)

def interference(cmd):
    nowtime = os.popen(cmd)
    print nowtime.read()

deci_ratio=2
Nsamples = 256
time_interval = 10
frequency_cut_off = 0.5
sampling_interval = 40000
q = Queue.LifoQueue()

t1 = Thread( target = work_flow,args=(deci_ratio,time_interval,"false","true",q,)) 
t2 = Thread( target = update_sampling,args=(sampling_interval, Nsamples, deci_ratio, time_interval, "false", frequency_cut_off, q,))
work_flow(deci_ratio,time_interval,"false","false",q)
#sampling("all_diff", Nsamples, "true")
#update_sampling(sampling_interval, Nsamples, deci_ratio, time_interval, "false", frequency_cut_off, q)
#t1.start()
#t2.start()
#fully_refine(deci_ratio,"false")
#blob_detection("original")
#t3 = Thread(target = fully_refine_n, args=(deci_ratio, 2000,))
#t4 = Thread(target = partial_refine_n, args=(deci_ratio, 360, "false", 1,2000,))
#fully_refine_n(deci_ratio, 2000)
#partial_refine_n(deci_ratio, 360, "false", 1,2000)
#t3.start()
#t4.start()

#t5 = Thread(target = fetch_from_HDD, args = (10000,))
#t6 = Thread(target = all_fetch_from_HDD, args = ("data",10000,))
#t5.start()
#t6.start()
#t7 = Thread(target = interference, args = ('./current/interference 512 20',))
#t8 = Thread(target = interference, args = ('./current/interference 512 24',))
#t9 = Thread(target = interference, args = ('./current/interference 512 26',))
#t10 = Thread(target = interference, args = ('./current/interference 512 28',))
#t7.start()
#t8.start()
#t9.start()
#t10.start()
#partial_refine(deci_ratio,140,"false",1,"true")
print "waiting\n"
#fetch_from_HDD_n(2000)
#all_fetch_from_HDD_n(2000)
#all_fetch_from_HDD("all_same",4000)
#all_fetch_from_HDD_n(1000)
#fetch_from_HDD(10000)
#all_fetch_from_HDD("data",100)
#fetch_from_HDD(10000)
#all_fetch_from_HDD("r","r","z",10000)
#write_chosn_data([0.1],[0],deci_ratio,1)
#partial_refine(deci_ratio, 0, "false", 1)
#reduced_data()
#plot_original()
#all_fetch_from_HDD("delta_L0_L1","delta_r_L0_L1","delta_z_L0_L1",10000)
#all_fetch_from_HDD("all_same","all_same","all_same",10000)


#try:
#	cluster = rados.Rados(conffile='')
#except TypeError as e:
#	print 'Argument validation error: ', e
#	raise e
#
#try:
#	cluster.connect()
#except Exception as e:
#	print "connection error: ", e
#	raise e
##finally:
##    print "Connected to the cluster."
#
#if not cluster.pool_exists('tier2_pool'):
#	raise RuntimeError('No data pool exists')
#ioctx_2 = cluster.open_ioctx('tier2_pool')
#
#dpot_L1_str = ioctx_2.read("dpot_L1",ioctx_2.stat("dpot_L1")[0],0)
#dpot_L1_o=struct.unpack(str(int(ioctx_2.stat("dpot_L1")[0]/8))+'d',dpot_L1_str)
#r_L1_str = ioctx_2.read("r_L1",ioctx_2.stat("r_L1")[0],0)
#r_L1_o=struct.unpack(str(int(ioctx_2.stat("r_L1")[0]/8))+'d',r_L1_str)
#z_L1_str = ioctx_2.read("z_L1",ioctx_2.stat("z_L1")[0],0)
#z_L1_o=struct.unpack(str(int(ioctx_2.stat("z_L1")[0]/8))+'d',z_L1_str)
#
#ioctx_2.close()
#cluster.shutdown()
#for i in range(len(dpot_L1)):
#    if dpot_L1[i]-dpot_L1_o[i] >0.01:
#        print dpot_L1[i],dpot_L1_o[i]
#    if r_L1[i]-r_L1_o[i] >0.01:
#        print r_L1[i],r_L1_o[i]
#    if z_L1[i]-z_L1_o[i] >0.01:
#        print z_L1[i],z_L1_o[i]
    
