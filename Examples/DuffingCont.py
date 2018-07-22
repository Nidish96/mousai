#!/usr/bin/python3

# %load_ext autoreload
# %autoreload 2

import mousai as ms
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.linalg as la
import ipdb
import importlib
importlib.reload(ms)

def fnlduffing(x, params):
    b = params['b']
    ft = b*x**3
    dfdxt = 3.0*b*x**2
    dfdxdt = np.zeros_like(x)
    dfdwt = np.zeros_like(x)
    return ft, dfdxt, dfdxdt, dfdwt

Forcinglevels = [ 0.25, 0.5, 0.75, 1, 1.25, 1.5 ];

params = dict()
m,c,k,b,fh = 1.0,0.1,1.0,1.0,1
params['M'] = np.reshape(m,(1,1))
params['C'] = np.reshape(c,(1,1))
params['K'] = np.reshape(k,(1,1))
params['b'] = np.reshape(b,(1,1))
params['fh'] = fh
params['Fc'] = np.reshape(0.50,(1,1))
params['Fs'] = np.reshape(0.00,(1,1))

Nh = 20;
Nt = 128;

Ws = 0.05
We = 4.00
ds = 0.01
dsmax = 0.10
dsmin = 0.00001
E = np.zeros((2*Nh+1,2*Nh+1))
E[0,0] = k
for n in range(1,Nh+1):
    E[2*n-1,2*n-1] = E[2*n,2*n] = k-(n*Ws)**2*m
    E[2*n-1,2*n] = (n*Ws)*c
    E[2*n,2*n-1] = -(n*Ws)*c
Fl = np.zeros((2*Nh+1,1))
Fl[2*fh-1] = 1.0
X0 = la.solve(E,Fl)
for F in Forcinglevels:
    params['Fc'] = np.reshape(F,(1,1))

    Xi = ms.hb_freq_cont(fnlduffing, X0=X0*F, Ws=Ws, We=We, ds=ds, deriv=True, fnform='Time', num_harmonics=Nh, eqform='second_order', params=params, num_time_points=Nt, solep=1e-10, ITMAX=100, scf=np.sqrt(2), Nop=6.0, dsmax=dsmax, dsmin=dsmin, zt=None, angop=10.0)

    W = Xi[-1,:]
    A = np.sqrt(Xi[1,:]**2 + Xi[2,:]**2)/np.sqrt(2)
    print(F)
    plt.plot(W,A,'o-')
plt.xlim(0.0, 4.5)
plt.show()
