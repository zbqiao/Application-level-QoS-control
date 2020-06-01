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
def find_large_elements(source,threshold):
    s_mean = np.mean(source)
    s_std = np.std(source,ddof=1)
    high = s_mean + 3.5 * s_std * threshold
    low = s_mean - 3.5 * s_std * threshold
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
    #cnt=0
    #for i in range(len(chosn_index_finer)):
    #    for j in range(len(chosn_index_finer[i])):
    #        cnt+=1   
    #print "number of chosn index=",cnt
    #print np.shape(chosn_index_finer)
    return chosn_index_finer

def refinement(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
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
#file=open("result1.txt","a")
def plot(data,r,z):
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
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=250));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig('w_finer.png', format='png')


start = time.time()
prefix_time=""
x=""
x_label = 0.0
deci_ratio=2
s_read_t_w=0
start = time.time()
for i in range(2000):
    #a=time.time()
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
    tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
    dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
    r_L1=struct.unpack(tag+'d',r_L1_str)
    z_L1=struct.unpack(tag+'d',z_L1_str)
    #chosn_b = time.time()
    chosn_index_group_L0_str = ioctx_0.read("chosn_index_group_L0",ioctx_0.stat("chosn_index_group_L0")[0],0)
    chosn_index_L0 = struct.unpack(str(int(ioctx_0.stat("chosn_index_group_L0")[0]/4))+'i',chosn_index_group_L0_str)
    finer_len_str=ioctx_0.get_xattr("chosn_index_group_L0", "finer_len")
    finer_len = int(finer_len_str)
    if chosn_index_L0 ==():
        finer_L0 = dpot_L1
        finer_r_L0 = r_L1
        finer_z_L0 = z_L1
    else:
        if not cluster.pool_exists('tier2_pool'):
    	    raise RuntimeError('No data pool exists')
        ioctx_2 = cluster.open_ioctx('tier2_pool')
        chosn_data_L0_str = ioctx_2.read("chosn_data_L0",ioctx_2.stat("chosn_data_L0")[0],0)
        chosn_r_L0_str = ioctx_2.read("chosn_r_L0",ioctx_2.stat("chosn_r_L0")[0],0)
        chosn_z_L0_str = ioctx_2.read("chosn_z_L0",ioctx_2.stat("chosn_z_L0")[0],0)
    
        tag=str(int(ioctx_2.stat("chosn_data_L0")[0]/8))
        chosn_data_L0 = struct.unpack(tag+'d',chosn_data_L0_str)
        chosn_r_L0 = struct.unpack(tag+'d',chosn_r_L0_str)   
        chosn_z_L0 = struct.unpack(tag+'d',chosn_z_L0_str)
        delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1",ioctx_2.stat("delta_r_L0_L1")[0],0)
        tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
        delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
        ioctx_2.close() 
        finer_L0, finer_r_L0,finer_z_L0 = refinement(chosn_index_L0, finer_len, chosn_data_L0,chosn_r_L0,chosn_z_L0,dpot_L1,r_L1,z_L1,deci_ratio)
    #print finer_r_L0
    #plot(finer_L0, finer_r_L0, finer_z_L0)
    #d = time.time()
    #s_read_t=d-b
    #print "refinement runtime=",d-b	
    ioctx_0.close()
    #print np.shape(dpot_L1)
    cluster.shutdown()
    #e=time.time()
    #print "performance on timestep %i = %f\n"%(i,e-a)
    #s_read_t_w+=s_read_t
    #prefix_time += str(e-a)+","
    #x_label += e-a 
    #x+=str(x_label)+","
end=time.time()
#file.write(prefix_time)
#file.write("\n---------------\n")
#file.write(x)
#file.close()
#print "single read time whole=",s_read_t_w

print "runtime=",end-start
#print len(dpot_L1)
#print len(finer_L0)
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

dpot_str = ioctx_0.read("data",ioctx_0.stat("data")[0],0)
r_str = ioctx_0.read("r",ioctx_0.stat("r")[0],0)
z_str = ioctx_0.read("z",ioctx_0.stat("z")[0],0)
tag=str(int(ioctx_0.stat("data")[0]/8))
dpot=struct.unpack(tag+'d',dpot_str)
r=struct.unpack(tag+'d',r_str)
z=struct.unpack(tag+'d',z_str)
data_len=[len(dpot_L1),len(dpot)]
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
psnr_finer_L0=psnr_c(dpot, finer_L0, data_len, deci_ratio, 1)
psnr_L0=psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
print "finer PSNR=",psnr_finer_L0
print "original PSNR=",psnr_L0


