import numpy as np
import nfft

# define evaluation points
x = -0.5 + np.random.rand(1000)

# define Fourier coefficients
N = 10000
k = N // 2 + np.arange(N)
f_k = np.random.randn(N)
print x
#print len(f_k)
# direct Fourier transform
#%time f_x_direct = nfft.ndft(x, f_k)

# fast Fourier transform
f_x_fast = nfft.nfft(x, f_k)

