n_groups = 4
 
# create plot
fig, ax = plt.subplots(figsize=(11,6))
#fig.set_size_inches(15, 10)
#plt.figure(figsize=(60,1)) 
index = np.arange(n_groups)
bar_width = 1.0/(3)
opacity = 0.8

ax.set_axisbelow(True)
ax.minorticks_on()

ax.yaxis.grid(which='major' , color='#d9d9d9',linestyle='-', linewidth=2, zorder=1)

ax.yaxis.grid(which='minor', color='#d9d9d9',linestyle='-',linewidth=0.5, zorder=2)
rects1 = plt.bar(index+2*bar_width, blob_number, bar_width, alpha=opacity, color='#4292c6', linewidth=0,zorder=30)
 
#rects2 = plt.bar(index + 2.5*bar_width, time_partial, bar_width, alpha=opacity, color='#d94801', label='Partial', linewidth=0, zorder=31)

#rects3 = plt.bar(index+3.5*bar_width, fully_psnr_m, bar_width, alpha=opacity, color='#41ab5d', label='Uniform', linewidth=0, zorder=32)
 
plt.xlabel('Different noise amplitude',fontsize=35)
plt.ylabel('Number of blobs',fontsize=30,labelpad=15)
#plt.title('Error percentage optimization (2X_mean)')
plt.xticks(index + 2*bar_width, ('0.8 (No refine)', '0.5', '0.1', '0 (Fully refine)'),fontsize=27)
plt.yticks(np.arange(0,21,5),fontsize=35)
#plt.legend(loc=9,fontsize=30,ncol=3,bbox_to_anchor=(0.5, 1.25),columnspacing=0.2,frameon=False)
#plt.legend(loc=0,ncol=2,mode="expand")
#plt.legend(loc=6, bbox_to_anchor=(1, 0.8))
#plt.yticks(list(plt.xticks()[0]) + [100])
plt.savefig('/Users/qzb/Dropbox/HPDC 2020/figure/N_blobs_dnoise.pdf', format='pdf',bbox_inches='tight')

n_groups = 4
 
# create plot
fig, ax = plt.subplots(figsize=(11,6))
#fig.set_size_inches(15, 10)
#plt.figure(figsize=(60,1)) 
index = np.arange(n_groups)
bar_width = 1.0/(3)
opacity = 0.8

ax.set_axisbelow(True)
ax.minorticks_on()

ax.yaxis.grid(which='major' , color='#d9d9d9',linestyle='-', linewidth=2, zorder=1)

ax.yaxis.grid(which='minor', color='#d9d9d9',linestyle='-',linewidth=0.5, zorder=2)
rects1 = plt.bar(index+2*bar_width, blob_dia, bar_width, alpha=opacity, color='#4292c6', linewidth=0,zorder=30)
 
#rects2 = plt.bar(index + 2.5*bar_width, time_partial, bar_width, alpha=opacity, color='#d94801', label='Partial', linewidth=0, zorder=31)

#rects3 = plt.bar(index+3.5*bar_width, fully_psnr_m, bar_width, alpha=opacity, color='#41ab5d', label='Uniform', linewidth=0, zorder=32)
 
plt.xlabel('Different noise amplitude',fontsize=35)
plt.ylabel('Avg. blob diameter (Pixel)',fontsize=25,labelpad=20)
#plt.title('Error percentage optimization (2X_mean)')
plt.xticks(index + 2*bar_width, ('0.8 (No refine)', '0.5', '0.1', '0 (Fully refine)'),fontsize=27)
plt.yticks(np.arange(0,21,5),fontsize=35)
#plt.legend(loc=9,fontsize=30,ncol=3,bbox_to_anchor=(0.5, 1.25),columnspacing=0.2,frameon=False)
#plt.legend(loc=0,ncol=2,mode="expand")
#plt.legend(loc=6, bbox_to_anchor=(1, 0.8))
#plt.yticks(list(plt.xticks()[0]) + [100])
plt.savefig('/Users/qzb/Dropbox/HPDC 2020/figure/dia_blobs_dnoise.pdf', format='pdf',bbox_inches='tight')


n_groups = 4
 
# create plot
fig, ax = plt.subplots(figsize=(11,6))
#fig.set_size_inches(15, 10)
#plt.figure(figsize=(60,1)) 
index = np.arange(n_groups)
bar_width = 1.0/(3)
opacity = 0.8

ax.set_axisbelow(True)
ax.minorticks_on()

ax.yaxis.grid(which='major' , color='#d9d9d9',linestyle='-', linewidth=2, zorder=1)

ax.yaxis.grid(which='minor', color='#d9d9d9',linestyle='-',linewidth=0.5, zorder=2)
rects1 = plt.bar(index+2*bar_width, blob_area, bar_width, alpha=opacity, color='#4292c6', linewidth=0,zorder=30)
 
#rects2 = plt.bar(index + 2.5*bar_width, time_partial, bar_width, alpha=opacity, color='#d94801', label='Partial', linewidth=0, zorder=31)

#rects3 = plt.bar(index+3.5*bar_width, fully_psnr_m, bar_width, alpha=opacity, color='#41ab5d', label='Uniform', linewidth=0, zorder=32)
 
plt.xlabel('Different noise amplitude',fontsize=35)
plt.ylabel('Aggr. blob area (Sqr pixel)',fontsize=25,labelpad=20)
#plt.title('Error percentage optimization (2X_mean)')
plt.xticks(index + 2*bar_width, ('0.8 (No refine)', '0.5', '0.1', '0 (Fully refine)'),fontsize=27)
plt.yticks(np.arange(0,7001,2000),fontsize=30)
#plt.legend(loc=9,fontsize=30,ncol=3,bbox_to_anchor=(0.5, 1.25),columnspacing=0.2,frameon=False)
#plt.legend(loc=0,ncol=2,mode="expand")
#plt.legend(loc=6, bbox_to_anchor=(1, 0.8))
#plt.yticks(list(plt.xticks()[0]) + [100])
plt.savefig('/Users/qzb/Dropbox/HPDC 2020/figure/area_blobs_dnoise.pdf', format='pdf',bbox_inches='tight')

n_groups = 4
 
# create plot
fig, ax = plt.subplots(figsize=(11,6))
#fig.set_size_inches(15, 10)
#plt.figure(figsize=(60,1)) 
index = np.arange(n_groups)
bar_width = 1.0/(3)
opacity = 0.8

ax.set_axisbelow(True)
ax.minorticks_on()

ax.yaxis.grid(which='major' , color='#d9d9d9',linestyle='-', linewidth=2, zorder=1)

ax.yaxis.grid(which='minor', color='#d9d9d9',linestyle='-',linewidth=0.5, zorder=2)
rects1 = plt.bar(index+2*bar_width, blob_overlap, bar_width, alpha=opacity, color='#4292c6', linewidth=0,zorder=30)
 
#rects2 = plt.bar(index + 2.5*bar_width, time_partial, bar_width, alpha=opacity, color='#d94801', label='Partial', linewidth=0, zorder=31)

#rects3 = plt.bar(index+3.5*bar_width, fully_psnr_m, bar_width, alpha=opacity, color='#41ab5d', label='Uniform', linewidth=0, zorder=32)
 
plt.xlabel('Different noise amplitude',fontsize=35)
plt.ylabel('Blob overlap ratio',fontsize=30,labelpad=20)
#plt.title('Error percentage optimization (2X_mean)')
plt.xticks(index + 2*bar_width, ('0.8 (No refine)', '0.5', '0.1', '0 (Fully refine)'),fontsize=27)
plt.yticks(np.arange(0,1.01,0.2),fontsize=30)
#plt.legend(loc=9,fontsize=30,ncol=3,bbox_to_anchor=(0.5, 1.25),columnspacing=0.2,frameon=False)
#plt.legend(loc=0,ncol=2,mode="expand")
#plt.legend(loc=6, bbox_to_anchor=(1, 0.8))
#plt.yticks(list(plt.xticks()[0]) + [100])
plt.savefig('/Users/qzb/Dropbox/HPDC 2020/figure/overlap_blobs_dnoise.pdf', format='pdf',bbox_inches='tight')

time=[67.7749750614,286.975735903, 305.683176994,317]
n_groups = 4
 
# create plot
fig, ax = plt.subplots(figsize=(11,6))
#fig.set_size_inches(15, 10)
#plt.figure(figsize=(60,1)) 
index = np.arange(n_groups)
bar_width = 1.0/(3)
opacity = 0.8

ax.set_axisbelow(True)
ax.minorticks_on()

ax.yaxis.grid(which='major' , color='#d9d9d9',linestyle='-', linewidth=2, zorder=1)

ax.yaxis.grid(which='minor', color='#d9d9d9',linestyle='-',linewidth=0.5, zorder=2)
rects1 = plt.bar(index+2*bar_width, time, bar_width, alpha=opacity, color='#4292c6', linewidth=0,zorder=30)
 
#rects2 = plt.bar(index + 2.5*bar_width, time_partial, bar_width, alpha=opacity, color='#d94801', label='Partial', linewidth=0, zorder=31)

#rects3 = plt.bar(index+3.5*bar_width, fully_psnr_m, bar_width, alpha=opacity, color='#41ab5d', label='Uniform', linewidth=0, zorder=32)
 
plt.xlabel('Different noise amplitude',fontsize=35)
plt.ylabel('time',fontsize=30,labelpad=20)
#plt.title('Error percentage optimization (2X_mean)')
plt.xticks(index + 2*bar_width, ('0.8 (No refine)', '0.5', '0.1', '0 (Fully refine)'),fontsize=27)
plt.yticks(np.arange(0,320,80),fontsize=30)
#plt.legend(loc=9,fontsize=30,ncol=3,bbox_to_anchor=(0.5, 1.25),columnspacing=0.2,frameon=False)
#plt.legend(loc=0,ncol=2,mode="expand")
#plt.legend(loc=6, bbox_to_anchor=(1, 0.8))
#plt.yticks(list(plt.xticks()[0]) + [100])
plt.savefig('/Users/qzb/Dropbox/HPDC 2020/figure/time_dnoise.pdf', format='pdf',bbox_inches='tight')


#y_mean = np.mean(y)
#y -= y_mean
f, Pxx_spec = scipy.signal.welch(y, sample_rate, 'flattop', 1024, scaling='spectrum')
#plt.figure(2,figsize=(11,9))
#plt.plot(f,np.sqrt(Pxx_spec))


#fig,ax = plt.subplots(1,figsize=(15,6))
plt.figure(1,figsize=(11,9))
plt.plot(x[:256],y[:256])


#y-=np.mean(y)
amp = fft.fft(y)/(Nsamples/2.0)

#amp = fft.fft(y)
#amp_t=fft.ifft(amp*(Nsamples/2.0))
#plt.figure(5,figsize=(11,9))
#plt.plot(x[:200],amp_t[:200])
#print amp[0]
amp_complex_h = amp[range(int(len(x)/2))]
amp_h = np.absolute(amp_complex_h)
#amp_temp=amp_h
#amp_sort=np.sort(amp_temp)
#print amp_sort
freq=fft.fftfreq(amp.size,1/sample_rate)
freq_h = freq[range(int(len(x)/2))]
#plt.figure(2,figsize=(11,9))
#plt.plot(freq, np.abs(amp))
#plt.plot(freq_h[:], amp_h[:])

hi_freq_ratio = 0.4

#print np.min(amp_h)
if amp_h[0]>1e-10:    
    threshold = np.max(np.delete(amp_h,0,axis=0))*hi_freq_ratio
    dc = amp_h[0]/2.0
    start_index = 1
else:
    threshold = np.max(amp_h)*hi_freq_ratio
    dc = 0.0
    start_index = 0
print "dc",dc
print "threshold",threshold
selected_freq = []
selected_amp = []
selected_complex=[]
#print start_index
for i in range(start_index,len(amp_h)):
#     i == 12:
    if amp_h[i]>=threshold:
        #print i
        selected_freq.append(freq_h[i])    
        selected_amp.append(amp_h[i])
        selected_complex.append(amp_complex_h[i])
#print selected_complex

#selected_phase = np.arctan(-np.array(selected_complex).real/np.array(selected_complex).imag)# get phase
selected_phase = np.arctan2(np.array(selected_complex).imag,np.array(selected_complex).real)
#print selected_phase
for i in range(len(selected_phase)):
    if np.fabs(selected_phase[i])<1e-10:
        selected_phase[i]=0.0
#print selected_freq
print selected_phase
#print selected_amp
print len(selected_freq)
Fs1 = sample_rate  
#Fs1=20
Ts1 = 1.0/Fs1 

# t = np.linspace(0,1,Fs) 
t = np.arange(0,x[-1], Ts1)
#t=5

sig = dc
for i in range(len(selected_freq)):
    sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*t+ selected_phase[i])
#print sig
#print noise_threshold(sig)
#sig = selected_amp[0]*np.sin(2*np.pi*selected_freq[0]*t+ selected_phase[0])
#print sig   
#fig,ax = plt.subplots(1,figsize=(15,6))
print "max=",np.max(sig)
plt.figure(1,figsize=(11,9))
plt.plot(t[:256], sig[:256])
#sig_t = dc
#t1 = x[-1]
#for i in range(start_index, len(selected_freq)):
#    sig_t += selected_amp[i]*np.sin(2*np.pi*selected_freq[i]*t1+ selected_phase[i])
#sig_t

y1=sig
Nsamples1=len(y1)
amp = fft.fft(y1)/(Nsamples1/2.0)
#amp = fft.fft(y)
#amp_t=fft.ifft(amp*(Nsamples/2.0))
#plt.figure(5,figsize=(11,9))
#plt.plot(x[:200],amp_t[:200])
#print amp[0]
amp_complex_h = amp[range(int(len(t)/2))]
amp_h = np.abs(amp_complex_h)
#amp_temp=amp_h
#amp_sort=np.sort(amp_temp)
#print amp_sort
freq=fft.fftfreq(amp.size,1/Fs1)
freq_h = freq[range(int(len(t)/2))]
#plt.figure(2,figsize=(11,9))
#plt.plot(freq, np.abs(amp))
#plt.plot(freq_h[:256], amp_h[:256])
plt.xlabel("First 256 steps time",fontsize=30)
plt.ylabel("Time",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.savefig("/Users/qzb/Dropbox/HPDC 2020/figure/high_amp_fft_2046_8.pdf",format='pdf',bbox_inches='tight')
#plt.savefig("/Users/qzb/Dropbox/HPDC 2020/figure/high_amp_fft_2046_8_1024_4.pdf",format='pdf',bbox_inches='tight')
