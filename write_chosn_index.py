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
    #print "uplimit=",uplimit
    return uplimit

def find_augment_points(base,chosn_index,deci_ratio):
    chosn_points=[]
    
    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    base_gradient=np.gradient(base)
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(math.fabs(base_gradient[j]))

    uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:

        for j in range(i[0],i[-1]+1):
            if math.fabs(base_gradient[j])>uplimit:
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
    print str(len(reduce(operator.add,chosn_index_finer))*100/len(base))+"%"

    #cnt=0
    #for i in range(len(chosn_index_finer)):
    #    for j in range(len(chosn_index_finer[i])):
    #        cnt+=1   
    #print "number of chosn index=",cnt
    #print np.shape(chosn_index_finer)
    return chosn_index_finer

def refinement(chosn_index, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros((len(base)-1)*deci_ratio+1)
    finer_r=np.zeros((len(base)-1)*deci_ratio+1)
    finer_z=np.zeros((len(base)-1)*deci_ratio+1)
    start_index = 0
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
    chosn_index_n=[]
    for i in chosn_index:
        chosn_index_n.append(range(i[0]*deci_ratio,i[-1]*deci_ratio+1))
    #print chosn_index
    chosn_index_1d=reduce(operator.add,chosn_index_n)
    n_selected_points=0
    start=0
    for i in range(len(chosn_index_n)):
        k=0
        for m in range(start,chosn_index_n[i][0]):
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
    
        for n in range(chosn_index_n[i][0],chosn_index_n[i][-1]+1):
            index1 = n // deci_ratio
            index2 = n % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    #print i,k
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[i][k]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[i][k]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[i][k]
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chson_data[i][k]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chson_r[i][k]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chson_z[i][k]
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            k+=1
        start = chosn_index_n[i][-1]+1
        if i == len(chosn_index_n)-1 and chosn_index_n[i][-1]!=(len(base)-1)*deci_ratio:
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
    #print "percentage of selected points= ", n_selected_points/(len(delta_L1)-len(base))
    #for i in chosn_index:
    #    if i[-1] != len(base)-1:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,(i[-1]+1)*deci_ratio))
    #    else:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,len(delta_L1)))
    #PSNR=psnr(delta_L1,chosn_index_n,np.max(finer),len(delta_L1))
    #print finer
    #print 3
    return finer, finer_r, finer_z, chosn_index_n

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

def k_means(source,savefig,k):
    if k == "true":
        find_k(source,savefig)
    y=np.array(source).reshape(-1,1)
    km=KMeans(n_clusters=100)
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

def find_k(source,savefig):
    SSE=[]
    scores=[]
    
    for i in range(2,10):
        estimator=KMeans(n_clusters=i)
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
     #   plt.savefig('/media/sf_Dropbox/HPDC 2020/figure/optimal_k.pdf',format='pdf',bbox_inches='tight')

start = time.time()

deci_ratio=2
for i in range(1):
    print i
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

    dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0])
    tag='d'
    for i in range(int(ioctx_0.stat("dpot_L1")[0]/8)-1):
        tag+='d'
    dpot_L1=struct.unpack(tag,dpot_L1_str)
    chosn_index_L0 = find_augment_points(dpot_L1, [range(len(dpot_L1))],deci_ratio)
    n_elems=0
    chosn_index_group_L1=[]
    for i in chosn_index_L0:
        chosn_index_group_L1.append(i[0] * deci_ratio)
        chosn_index_group_L1.append(i[-1] * deci_ratio)

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
    
    chosn_data_L1=[]
    chosn_r_L1=[]
    chosn_z_L1=[]
    for i in chosn_index_L0:
        for j in range(i[0]*deci_ratio, i[-1]*deci_ratio+1):
            chosn_data_L1.append(delta_L0_L1[j])
            chosn_r_L1.append(delta_r_L0_L1[j])
            chosn_z_L1.append(delta_z_L0_L1[j])  
    print len(chosn_data_L1)  
    #group_length=[]
    #for i in chosn_index_L0:
    #    group_length.append(len(i))
    #chosn_index_L0_1d=reduce(operator.add,chosn_index_L0) 
    #print len(chosn_index_L0_1d)
    ioctx_0.write_full("chosn_index_group_L1",struct.pack(str(len(chosn_index_group_L1))+'i',*chosn_index_group_L1))
    ioctx_2.write_full("chosn_data_L1",struct.pack(str(len(chosn_data_L1))+'d',*chosn_data_L1))
    ioctx_2.write_full("chosn_r_L1",struct.pack(str(len(chosn_r_L1))+'d',*chosn_r_L1))
    ioctx_2.write_full("chosn_z_L1",struct.pack(str(len(chosn_z_L1))+'d',*chosn_z_L1))
    #ioctx_0.write_full("group_length_L0",struct.pack(str(len(group_length))+'i',*group_length))
    #ioctx_0.set_xattr("dpot_L1","chosn_index_L0",struct.pack(str(n_elems)+'d',*chosn_index_L0_1d))
    #ioctx_0.set_xattr("dpot_L1","group_length_L0",struct.pack(str(len(group_length))+'i',*group_length))
    #print len(reduce(operator.add,chosn_index_L0))
    #print len(dpot_L1)
    
    
    ioctx_0.close()
    ioctx_2.close()
    #print np.shape(dpot_L1)
    cluster.shutdown()
end=time.time()
print "runtime=",end-start

