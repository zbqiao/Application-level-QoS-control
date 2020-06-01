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
from scipy.fftpack import fft

np.set_printoptions(threshold=np.inf)
def find_large_elements(source,threshold):
    s_mean = np.mean(source)
    s_std = np.std(source,ddof=1)
    high = s_mean + 4 * s_std * threshold
    low = s_mean - 4 * s_std * threshold
    return high,low

def find_augment_points(base,chosn_index,deci_ratio,threshold):
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
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if base_gradient[j]>high_b or base_gradient[j]<low_b:
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
    #print "percentage of selected points= ", n_selected_points/(len(delta_L1)-len(base))
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
    km=KMeans(n_clusters=4)
    km.fit(y)
    km_label=km.labels_
    #print "source=",source
    #print "km.label=",km_label
    #print km.cluster_centers_
    if len(km_label)!=len(source):
        print "length issue"
    sorted_cluster_index=np.argsort(km.cluster_centers_.reshape(-1,))
    #print "sorted_cluster_index=",sorted_cluster_index
    group=sorted_cluster_index[:2]
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

def preprocess_noise(y):
    first_tag=1
    first_noise_tag =1
    temp=[]
    delete=[]
    noise_len=[]
    noise_index =[]
    noise_line =0.2
    for i in range(len(y)):
        if y[i] > noise_line:
            if first_tag == 1:
                last_noise = i
                temp.append(i)
                first_tag=0
                continue
            else:
                if i - last_noise < sample_rate * 6:
                    temp.append(i)
                else:
                    #print temp
                    for m in range(len(temp)):
                        if y[temp[m]]<0.25 and m!=0 and m!=len(temp)-1:
                            delete.append(temp[m])
                    if first_noise_tag==0:
                        noise_index.append([temp[0],temp[-1]])

                    for n in range(len(delete)):
                        temp.remove(delete[n])
                    delete=[]

                        #print temp[-1]-temp[0]+1
                    for j in range(len(temp)-1):
                        for k in range(1,temp[j+1]-temp[j]):
                            if y[temp[j+1]]>y[temp[j]]:
                                y[temp[j]+k]=y[temp[j]]+(y[temp[j+1]]-y[temp[j]])*k/(temp[j+1]-temp[j])
                            else:
                                y[temp[j]+k]=y[temp[j+1]]+(y[temp[j]]-y[temp[j+1]])*k/(temp[j+1]-temp[j])
                    temp=[]

                    #print "i=",i
                    temp.append(i)
                    first_noise_tag=0
                last_noise=i 

    for j in range(len(temp)-1):
        for k in range(1,temp[j+1]-temp[j]):
            if y[temp[j+1]]>y[temp[j]]:
                y[temp[j]+k]=y[temp[j]]+(y[temp[j+1]]-y[temp[j]])*k/(temp[j+1]-temp[j])
            else:
                y[temp[j]+k]=y[temp[j+1]]+(y[temp[j]]-y[temp[j+1]])*k/(temp[j+1]-temp[j]) 
    return y,noise_index

def noise_threshold(noise):
    return 0.3

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
#file=open("result_fully.txt","a")
#prefix_time=""
#x=""
#x_label = 0.0
start = time.time()
#s_refine_t_w=0
#time_single=[]
#time_accumulate=[]
#subprocess.Popen(['./interference', '64', '1'])
#print "data finish\n"
#sample_rate = 2
#Nsamples = 512
for i in range(1000):
    a =time.time()
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
    #print ioctx_2.stat("delta_L0_L1")[0]
    #before = time.time()
    delta_L0_L1_str = ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0],0)
    delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
    delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0],0)
    #after = time.time()
    #print "after - before", after- before
    tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
    delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
    delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
    delta_z_L0_L1 = struct.unpack(tag+'d',delta_z_L0_L1_str)
    #b=time.time()
    #print "read and decode time=",b-a
    w_finer, w_finer_r,w_finer_z = whole_refine(dpot_L1,r_L1,z_L1,delta_L0_L1,delta_r_L0_L1,delta_z_L0_L1,2)
    #c=time.time()
    #s_refine_t=c-b
    #print "refinement time=",c-b
    
    #print "w_finer_r=",w_finer_r
    #print "w_finer_z=",w_finer_z
    #plot(w_finer, w_finer_r,w_finer_z) 

    #ioctx_0.close()
    ioctx_2.close()
    cluster.shutdown()
    #s_refine_t_w+=s_refine_t
    e=time.time()
    #time_single.append(e-a)
    #prefix_time += str(e-a)+","
    #x_label += e-a
    #time_accumulate.append(x_label)
    #x+=str(x_label)+","
    #if 1/sample_rate-(e-a)>0.0:
    #    time.sleep(1/sample_rate-(e-a))
    #else:
    #    print "Chosse another sample rate, it is too large!\n"
end=time.time()
print "runtime=",end-start
