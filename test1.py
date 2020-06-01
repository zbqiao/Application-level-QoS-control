import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

np.random.seed(12345)


# function we want to reconstruct
k=[1,5,10,30] # modulating coefficients
def myf(x,k): 
    return 10 + sum(np.sin(x*k0*(2*np.pi)) for k0 in k)

x=np.linspace(-0.5,0.5,1000)   # 'continuous' time/spatial domain; -0.5<x<+0.5
y=myf(x,k)                     # 'true' underlying trigonometric function
print len(y)


                        # we should sample at a rate of >2*~max(k)
M=256                   # number of nodes
N=256                   # number of Fourier coefficients

nodes =np.random.rand(M)-0.5 # non-uniform oversampling
values=myf(nodes,k)     # nodes&values will be used below to reconstruct 
                        # original function using the Solver


import numpy as np
from pynfft import NFFT, Solver
values-=np.mean(values)
f     = np.empty(M,     dtype=np.complex128)
f_hat = np.empty([N,N], dtype=np.complex128)

this_nfft = NFFT(N=[N,N], M=M)
this_nfft.x = np.array([[node_i,0.] for node_i in nodes])
#%pylab inline --no-import-all 
#%pylab inline --no-import-all 
this_nfft.precompute()

this_nfft.f = f
ret2=this_nfft.adjoint()

print this_nfft.M  # number of nodes, complex typed
print this_nfft.N  # number of Fourier coefficients, complex typed
#print this_nfft.x # nodes in [-0.5, 0.5), float typed


this_solver = Solver(this_nfft)
this_solver.y = np.array(values)          # '''right hand side, samples.'''

#this_solver.f_hat_iter = f_hat # assign arbitrary initial solution guess, default is 0

this_solver.before_loop()       # initialize solver internals
print this_solver.r_iter
#while not np.all(this_solver.r_iter < 1e-2):
#    print "bobo\n"
#    this_solver.loop_one_step()
niter = 1
for iiter in range(niter):
    this_solver.loop_one_step()

fig=plt.figure(1,(20,5))
ax =fig.add_subplot(111)

foo=[ np.abs( this_solver.f_hat_iter[i][0])**2 for i in range(len(this_solver.f_hat_iter) ) ]

ax.plot(np.abs(np.arange(-N/2,+N/2,1)),foo)

plt.savefig('test1.pdf', format='pdf',bbox_inches='tight') 
