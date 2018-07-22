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

def fnl2dofduff(x, params):
    b = params['b']
    ft = np.block([b*x[:,0:1]**3,b*x[:,0:1]**2*x[:,1:2]])
    dfdxt = np.block([3.0*b*x[:,0:1]**2,np.zeros_like(x[:,0:1]),2*b*x[:,0:1]*x[:,1:2],b*x[:,0:1]**2])
    dfdxdt = np.zeros_like(dfdxt)
    dfdwt = np.zeros_like(ft)
    return ft, dfdxt, dfdxdt, dfdwt

nd = 2;

params = dict()
m,c,k,b,fh = 1.0,0.1,1.0,1.0,1
M = np.array([[m,0.0],[0.0,m]])
C = np.array([[c,0.0],[0.0,c]])
K = np.array([[2*k,-k],[-k,k]])
Fc = np.reshape(np.array([0.0,1.4]),(nd,1))
params['M'] = M
params['C'] = C
params['K'] = K
params['b'] = np.reshape(b,(1,1))
params['fh'] = fh
params['Fc'] = Fc
params['Fs'] = np.zeros_like(Fc)


Nh = 2;
Nt = 128;

Ws = 0.01
We = 4.00
ds = 0.1
dsmax = 0.2;

E = np.zeros((nd*(2*Nh+1),nd*(2*Nh+1)))
E[0:nd,0:nd] = K
for n in range(1,Nh+1):
    cst = nd + (n-1)*2*nd
    cen = nd + (n-1)*2*nd + nd
    sst = nd + (n-1)*2*nd + nd
    sen = nd + (n-1)*2*nd + 2*nd
    E[cst:cen,cst:cen] = E[sst:sen,sst:sen] = K - (n*Ws)**2*M
    E[cst:cen,sst:sen] = (n*Ws)*C
    E[sst:sen,cst:cen] = -(n*Ws)*C
Fl = np.zeros((nd*(2*Nh+1),1))
Fl[nd+(fh-1)*(2*nd):2*nd+(fh-1)*(2*nd),:] = Fc
X0 = la.solve(E,Fl)

Xi = ms.hb_freq_cont(fnl2dofduff, X0=X0, Ws=Ws, We=We, ds=ds, deriv=True, fnform='Time', num_harmonics=Nh, eqform='second_order', params=params, num_time_points=Nt, solep=1e-10, ITMAX=100, dsmax=dsmax)
W=Xi[-1,:]
A1 = np.sqrt(Xi[2,:]**2+Xi[4,:]**2)
A2 = np.sqrt(Xi[3,:]**2+Xi[5,:]**2)
plt.plot(W,A1)
plt.plot(W,A2)
plt.show()

# for F in Forcinglevels:
#     params['Fc'] = np.reshape(F,(1,1))

#     E = np.zeros((2*Nh+1,2*Nh+1))
#     E[0,0] = k
#     for n in range(1,Nh+1):
#         E[2*n-1,2*n-1] = E[2*n,2*n] = k-(n*Ws)**2*m
#         E[2*n-1,2*n] = (n*Ws)*c
#         E[2*n,2*n-1] = -(n*Ws)*c
#     Fl = np.zeros((2*Nh+1,1))
#     Fl[2*fh-1]=F
#     X0 = la.solve(E,Fl)

#     Xi = ms.hb_freq_cont(fnlduffing, X0=X0, Ws=Ws, We=We, ds=ds, deriv=True, fnform='Time', num_harmonics=Nh, eqform='second_order', params=params, num_time_points=Nt, solep=1e-10, ITMAX=100, dsmax=0.50, scf=np.sqrt(2))

#     W = Xi[-1,:]
#     A = np.sqrt(Xi[1,:]**2 + Xi[2,:]**2)/np.sqrt(2)
#     print(F)
#     plt.plot(W,A)
# plt.xlim(0.0, 4.5)
# plt.show()
