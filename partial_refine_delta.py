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
from sklearn.cluster import KMeans
import operator
from functools import reduce
from sklearn.metrics import silhouette_score

np.set_printoptions(threshold=np.inf)

def quantile(source, k):
    percentile = np.percentile(source,[25,75])
    IQR = percentile[1]-percentile[0]
    uplimit = percentile[1]+IQR*k
    print "uplimit=",uplimit
    return uplimit

def find_k(source,savefig):
    SSE=[]
    scores=[]
    
    for i in range(2,10):
        estimator=KMeans(n_clusters=i)
        print i, np.array(source).reshape(-1,1)
        estimator.fit(np.array(source).reshape(-1,1))
        SSE.append(estimator.inertia_)
        #print estimator.labels_
        scores.append(silhouette_score(np.array(source).reshape(-1,1),estimator.labels_,metric='euclidean'))
    #print SSE
    X=range(2,10)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xticks(fontsize=20)
    ax1.plot(X,SSE,'ro-')
    #ax1.set_yticks(range(0,85000,20000))
    ax1.set_ylabel('SSE',fontsize=20)
    #ax1.set_yticklabels(['0','20000','40000','60000','80000'],fontsize=20)
    ax2 = ax1.twinx()
    ax2.plot(X,scores,'bD-')
    ax2.set_ylabel('Silhouette coefficient',fontsize=20)
    ax2.set_xlabel('k')
    #ax2.set_yticks([0.55,0.65,0.75,0.85])
    #ax2.set_yticklabels(['0.55','0.65','0.75','0.85'],fontsize=20)
    #ax2.set_xticklabels(['2','3','4','5','6','7','8','9'],fontsize=30)
    #plt.show()
    #if savefig=='true':
        #plt.savefig('/media/sf_Dropbox/HPDC 2020/figure/optimal_k.pdf',format='pdf',bbox_inches='tight')

def k_means(source,savefig,k):
    if k=='true':
        find_k(source,savefig)
    y=np.array(source).reshape(-1,1)
    km=KMeans(n_clusters=4)
    km.fit(y)
    km_label=km.labels_
    
    #print "km.label=",km_label
    #print km.cluster_centers_
    if len(km_label)!=len(source):
        print "length issue"
    sorted_cluster_index=np.argsort(km.cluster_centers_.reshape(-1,))
    print "sorted_cluster_index=",sorted_cluster_index
    group=sorted_cluster_index[:3]
    print "group=",group
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

def find_augment_delta_m(delta_L2,chosn_index,deci_ratio):
    chosn_delta=[]
    
    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    #print len(delta_L2)
    #print chosn_index
    for i in chosn_index:
        #print i[0]*deci_ratio, i[-1]*deci_ratio+1
        #print range(i[0]*deci_ratio,i[-1]*deci_ratio+1)
        for j in range(i[0]*deci_ratio,i[-1]*deci_ratio+1):
            #print j
            if delta_L2[j]!=0.0:
                #
                chosn_delta.append(math.fabs(delta_L2[j]))
                #chosn_delta.append(delta_L2[j])
        if i[-1] ==(len(delta_L2)-1)//deci_ratio and (len(delta_L2)-1)%deci_ratio!=0:
            for k in range(i[-1]*deci_ratio+1,i[-1]*deci_ratio+(len(delta_L2)-1)%deci_ratio+1):
                chosn_delta.append(math.fabs(delta_L2[k]))  
                #chosn_delta.append(delta_L2[k])
    #print "chosn_delta_m=", chosn_delta
    uplimit = quantile(chosn_delta,1.5)   
    #uplimit = k_means(chosn_delta,'false')
    #uplimit=outlier(chosn_delta)
    print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0]*deci_ratio,i[-1]*deci_ratio+1):
            if math.fabs(delta_L2[j])>uplimit:
                #print "large"
                temp_1.append(j)
        if i[-1] ==(len(delta_L2)-1)//deci_ratio and (len(delta_L2)-1)%deci_ratio!=0:
            for k in range(i[-1]*deci_ratio+1,i[-1]*deci_ratio+(len(delta_L2)-1)%deci_ratio+1):
                if math.fabs(delta_L2[k])>uplimit:
                    temp_1.append(k)
        if len(temp_1)>1:
            temp_index.append(temp_1)
        temp_1=[]
    #print temp_index                                      
    for i in temp_index:
        for j in range(1,len(i)):
            if i[j]-i[j-1]>1:
                temp_interval.append(i[j]-i[j-1])
    print temp_interval
    max_intv = k_means(temp_interval,'false','false')
    
    #max_intv=quantile(temp_interval,1)
    print "max_intv=",max_intv
    #print temp_index                  
    temp_2=[]  
#    cnt=0
    for i in temp_index:
        temp_2.append(i[0])
        for j in range(1,len(i)):
            if i[j]-i[j-1] <= max_intv:
                temp_2.append(i[j])
            else:
                #print temp_2
                if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
#                 for m in range(temp_2[0],temp_2[-1]+1):
#                     if m%deci_ratio!=0:
#                         cnt+=1
                temp_2=[]
                temp_2.append(i[j])
        if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
#         for n in range(temp_2[0],temp_2[-1]+1):
#             if n%deci_ratio!=0:
#                 cnt+=1
        temp_2=[]
#     if len(delta_L2)%deci_ratio!=0:
#         base_len=len(delta_L2)//deci_ratio+1
#     else:
#         base_len=len(delta_L2)//deci_ratio
#     print "Percentage of points used:",cnt/(len(delta_L2)-base_len)    
    #print chosn_index_finer
    return chosn_index_finer

def refinment_new(chosn_index, base, base_r, base_z, delta_L1, delta_L1_r, delta_L1_z, deci_ratio):
    finer=np.zeros(len(delta_L1))
    finer_r=np.zeros(len(delta_L1))
    finer_z=np.zeros(len(delta_L1))
    #finer_index=[]
    start_index = 0
    index_map={}
    a=0
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
    #print chosn_index
    chosn_index_n=reduce(operator.add,chosn_index)
    n_selected_points=0
    for i in range(len(delta_L1)):
        index1 = i//deci_ratio
        index2 = i%deci_ratio
        if i not in chosn_index_n:
            #print "Not in chosn_index_n"
            if index1!=len(base)-1:
                if index2!=0:
                    finer[i]=(base[index1]+base[index1+1])*index2/deci_ratio
                    finer_r[i]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                    finer_z[i]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                else:
                    finer[i]=base[index1]
                    finer_r[i]=base_r[index1]
                    finer_z[i]=base_z[index1]
                    finer_p.append(base[index1])
                    finer_r_p.append(base_r[index1])
                    finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[i]=2*base[index1]*index2/deci_ratio
                    finer_r[i]=2*base_r[index1]*index2/deci_ratio
                    finer_z[i]=2*base_z[index1]*index2/deci_ratio
                else:
                    finer[i]=base[index1]
                    finer_r[i]=base_r[index1]
                    finer_z[i]=base_z[index1]
                    finer_p.append(base[index1])
                    finer_r_p.append(base_r[index1])
                    finer_z_p.append(base_z[index1])
                    
        elif i in chosn_index_n:
            #print "In chosn_index_n"
            if index1!= len(base)-1:
                if index2 != 0:
                    finer[i]=(base[index1]+base[index1+1])*index2/deci_ratio+delta_L1[i]
                    finer_r[i]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+delta_L1_r[i]
                    finer_z[i]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+delta_L1_z[i]
                    finer_p.append((base[index1]+base[index1+1])*index2/deci_ratio+delta_L1[i])
                    finer_r_p.append((base_r[index1]+base_r[index1+1])*index2/deci_ratio+delta_L1_r[i])
                    finer_z_p.append((base_z[index1]+base_z[index1+1])*index2/deci_ratio+delta_L1_z[i])
                    n_selected_points+=1
                else:
                    finer[i]=base[index1]
                    finer_r[i]=base_r[index1]
                    finer_z[i]=base_z[index1]
                    finer_p.append(base[index1])
                    finer_r_p.append(base_r[index1])
                    finer_z_p.append(base_z[index1])
            else:
                if index2 != 0:
                    finer[i]=(base[index1]*2)*index2/deci_ratio+delta_L1[i]
                    finer_r[i]=(base_r[index1]*2)*index2/deci_ratio+delta_L1_r[i]
                    finer_z[i]=(base_z[index1]*2)*index2/deci_ratio+delta_L1_z[i]
                    finer_p.append((base[index1]*2)*index2/deci_ratio+delta_L1[i])
                    finer_r_p.append((base_r[index1]*2)*index2/deci_ratio+delta_L1_r[i])
                    finer_z_p.append((base_z[index1]*2)*index2/deci_ratio+delta_L1_z[i])
                    n_selected_points+=1
                else:
                    finer[i]=base[index1]
                    finer_r[i]=base_r[index1]
                    finer_z[i]=base_z[index1]
                    finer_p.append(base[index1])
                    finer_r_p.append(base_r[index1])
                    finer_z_p.append(base_z[index1])
    print "percentage of selected points= ", n_selected_points/(len(delta_L1)-len(base))
    #PSNR=psnr(delta_L1,chosn_index_n,np.max(finer),len(delta_L1))
    #print finer
    return finer, finer_r, finer_z,finer_p,finer_r_p,finer_z_p

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
        #print i, original_data[i]-leveldata[i]
        MSE=(original_data[i]-base[i])*(original_data[i]-base[i])+MSE
    MSE=MSE/len(original_data)
    psnr=10*math.log(np.max(original_data)**2/MSE,10)
    #print "psnr=",psnr
    return psnr

start = time.time()
for i in range(5000):
    print i
    try:
        cluster = rados.Rados(conffile='')
    except TypeError as e:
        print 'Argument validation error: ', e
        raise e

    print "Created cluster handle."

    try:
        cluster.connect()
    except Exception as e:
        print "connection error: ", e
        raise e
    finally:
        print "Connected to the cluster."

    print "I/O Context and Object Operations"
    print "================================="

    print "Creating a context  pool"

    if not cluster.pool_exists('tier0_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_0 = cluster.open_ioctx('tier0_pool')

    dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0])
    r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0])
    z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0])

    tag='d'
    for i in range(int(ioctx_0.stat("dpot_L1")[0]/8)-1):
        tag+='d'
    dpot_L1=struct.unpack(tag,dpot_L1_str)
    r_L1=struct.unpack(tag,r_L1_str)
    z_L1=struct.unpack(tag,z_L1_str)

    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')

    delta_L0_L1_str = ioctx_2.read("delta_L0_L1",ioctx_2.stat("delta_L0_L1")[0])
    delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0])
    delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1",ioctx_2.stat("delta_z_L0_L1")[0])

	tag='d'
        for i in range(int(ioctx_2.stat("delta_L0_L1")[0]/8)-1):
                tag+='d'
        delta_L0_L1 = struct.unpack(tag,delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack(tag,delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack(tag,delta_z_L0_L1_str)

    w_finer, w_finer_r,w_finer_z = whole_refine(dpot_L1,r_L1,z_L1,delta_L0_L1,delta_r_L0_L1,delta_z_L0_L1,2)
    #print "w_finer_r=",w_finer_r
    #print "w_finer_z=",w_finer_z
    #plot(w_finer, w_finer_r,w_finer_z) 

    print "Closing the connection."
    ioctx_0.close()
    ioctx_2.close()
    print np.shape(dpot_L1)
    print "Shutting down the handle."
    cluster.shutdown()
end=time.time()
print "runtime=",end-start	
