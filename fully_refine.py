# -*- coding: utf-8 -*-
from __future__ import division
import rados
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

np.set_printoptions(threshold=np.inf)
def find_large_elements(source,threshold):
    s_mean = np.mean(source)
    s_std = np.std(source,ddof=1)
    high = s_mean + 3.5 * s_std * threshold
    low = s_mean - 3.5 * s_std * threshold
    return high,low

def find_augment_points(base,chosn_index,deci_ratio,threshold):
    if threshold == 0.0:
        return [range(len(base))]
    elif threshold == 1.0:
        return []
    chosn_points=[]

    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    base_gradient=np.gradient(base)
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(base_gradient[j])
   
    high_b, low_b = find_large_elements(chosn_points, threshold)
    #print high_b, low_b
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if base_gradient[j]>=high_b or base_gradient[j]<low_b:
                #print j
                temp_1.append(j)
        if len(temp_1)>1:
            temp_index.append(temp_1)
        temp_1=[]
                                          
    for i in temp_index:
        for j in range(1,len(i)):
            if i[j]-i[j-1]>1:
                temp_interval.append(i[j]-i[j-1]) 
    
    #print "temp_interval",temp_interval
    max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    print "max_intv=",max_intv
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

def partial_refinement(chosn_index, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
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
    if noise > 0.75:
        threshold = 0.0
    elif noise < 0.1:
        threshold = 1.0 
    else:
        k=(1.0-0.0)/(0.1-0.75)
        b =1.0-k*0.1
        threshold = noise*k+b

    return threshold

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def plot(data,r,z):
    points = np.transpose(np.array([z,r]))
    print points
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices

	#plt.figure(206, figsize=(10,7))
    fig,ax=plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=26)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  

    axis_font = {'fontname':'Arial', 'size':'38'}

    #plt.xlabel('R', **axis_font)
   #plt.ylabel('Z',**axis_font )
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=250));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    	    spine.set_visible(False)
    plt.savefig('w_finer.png', format='png')

def whole_refine(base,base_r,base_z,delta_L1,delta_L1_r,delta_L1_z,deci_ratio):
    finer=np.zeros(len(delta_L1))
    finer_r=np.zeros(len(delta_L1))
    finer_z=np.zeros(len(delta_L1))

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
    return finer,finer_r,finer_z

def sampling(small_block_name, Nsamples, ioctx_2):
    prefix_time=""
    y=[] 
    for i in range(Nsamples):
        a =time.time()

        test_sample_str = ioctx_2.read(small_block_name, ioctx_2.stat("test_sample")[0],0)
    #tag = str(int(ioctx_2.stat("test_sample")[0]/8))
    #delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)

        e=time.time()
    #print e-a
        y.append(e-a)

    for i in range(len(y)-1):
        prefix_time += str(y[i])+","
    prefix_time += str(y[-1])
    
    file=open("result_fully.txt","a")
    file.write(prefix_time)
    file.close()
   
    x=[]
	x1=0
	for i in y:
    	x1+=i
    	x.append(x1)
   
    return x,y

#sample_rate = 2
Nsamples = 1024
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

dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
r_L1=struct.unpack(tag+'d',r_L1_str)
z_L1=struct.unpack(tag+'d',z_L1_str)
ioctx_0.close()

if not cluster.pool_exists('tier2_pool'):
	raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('tier2_pool')
delta_L0_L1_str = ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0],0)
delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0],0)
tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
delta_z_L0_L1 = struct.unpack(tag+'d',delta_z_L0_L1_str)

w_finer, w_finer_r,w_finer_z = whole_refine(dpot_L1,r_L1,z_L1,delta_L0_L1,delta_r_L0_L1,delta_z_L0_L1,2)
if steps % 10 ==0
	x,y = sampling("test_sample", Nsamples, ioctx_2):

ioctx_2.close()
cluster.shutdown()

#print "runtime=",end-start

print len(y)
sample_rate = Nsamples/x[-1]
amp = fft.fft(y)/(Nsamples/2.0)
amp_complex_h = amp[range(int(len(x)/2))]
amp_h = np.absolute(amp_complex_h)

freq=fft.fftfreq(amp.size,1/sample_rate)
freq_h = freq[range(int(len(x)/2))]

hi_freq_ratio = 0.4

if amp_h[0]>1e-10:    
    threshold = np.max(np.delete(amp_h,0,axis=0))*hi_freq_ratio
    dc = amp_h[0]/2.0
    start_index = 1
else:
    threshold = np.max(amp_h)*hi_freq_ratio
    dc = 0.0
    start_index = 0
print "dc",dc
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
Fs1 = sample_rate  
Ts1 = 1.0/Fs1 

# t = np.linspace(0,1,Fs) 
#t = np.arange(0,x[-1], Ts1)
sig = dc
future_t = 6.5
#future_t = 2.9  
for i in range(len(selected_freq)):
    sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*future_t+ selected_phase[i])
print "Noise amplitude=",sig
threshold1 = noise_threshold(sig)
#threshold1 = 0.0
print "Points selection threshold=",threshold1
deci_ratio=2
chosn_index_L1 = find_augment_points(dpot_L1,[range(len(dpot_L1))],deci_ratio, 1-threshold1)
#print len(chosn_index_L1[0])
if threshold1  == 0.0:
    chosn_index_group_L0=[]
elif threshold1 == 1.0:
    chosn_index_group_L0=[0,len(delta_L0_L1)-1]
else:
    chosn_index_group_L0=[] 
    for i in range(len(chosn_index_L1)):
        chosn_index_group_L0.append(chosn_index_L1[i][0] * deci_ratio)
        chosn_index_group_L0.append(chosn_index_L1[i][-1] * deci_ratio)
#print "chosn_index_group_L0=",chosn_index_group_L0

chosn_data_L0=[]
chosn_r_L0=[]
chosn_z_L0=[]
if threshold1 ==1.0:
    chosn_data_L0 = delta_L0_L1
    chosn_r_L0 = delta_r_L0_L1
    chosn_z_L0 = delta_z_L0_L1
else:
    for i in chosn_index_L1:
        for j in range(i[0]*deci_ratio, i[-1]*deci_ratio+1):
            chosn_data_L0.append(delta_L0_L1[j])
            chosn_r_L0.append(delta_r_L0_L1[j])
            chosn_z_L0.append(delta_z_L0_L1[j])
#print len(chosn_data_L0)
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
#print "Connected to the cluster."
if not cluster.pool_exists('tier0_pool'):
	raise RuntimeError('No data pool exists')
ioctx_0 = cluster.open_ioctx('tier0_pool')

if not cluster.pool_exists('tier2_pool'):
    raise RuntimeError('No data pool exists')

print "Points selection percentage =",len(chosn_data_L0)/len(delta_L0_L1)
#print chosn_index_group_L0
ioctx_2 = cluster.open_ioctx('tier2_pool')
ioctx_0.write_full("chosn_index_group_L0",struct.pack(str(len(chosn_index_group_L0))+'i',*chosn_index_group_L0))
ioctx_0.set_xattr("chosn_index_group_L0", "finer_len", str(len(delta_L0_L1)))
ioctx_2.write_full("chosn_data_L0",struct.pack(str(len(chosn_data_L0))+'d',*chosn_data_L0))
ioctx_2.write_full("chosn_r_L0",struct.pack(str(len(chosn_r_L0))+'d',*chosn_r_L0))
ioctx_2.write_full("chosn_z_L0",struct.pack(str(len(chosn_z_L0))+'d',*chosn_z_L0))
ioctx_0.close()
ioctx_2.close()
cluster.shutdown()
                                                                                    
