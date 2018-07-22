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

def ElDryFrict(Xn,Xt,params):
    mu = params['mu']
    kt = params['kt']
    kn = params['kn']
    Nh = params['Nh']
    Nt = params['Nt']
    soleps = params['eps']
    t = np.linspace(0,2*np.pi,Nt,endpoint=0)
    t = np.reshape(t,(t.size,1))
    xt = ms.freq2time(Xt, Nt)
    xn = ms.freq2time(Xn, Nt)
    
    fn = np.max(np.block([kt*xn,np.zeros_like(xn)]),axis=1)
    fn = np.reshape(fn,(Nt,1))
    cflg = np.sign(fn)
    dfndxn = np.zeros((Nt,2*Nh+1))
    dfndxn[:,0:1] = kn*cflg
    for n in range(1,Nh+1):
        dfndxn[:,(2*n-1):(2*n)] = dfndxn[:,0:1]*np.cos(n*t)
        dfndxn[:,(2*n):(2*n+1)] = dfndxn[:,0:1]*np.sin(n*t)

    ft = np.zeros_like(xt)
    dftdxt = np.zeros((Nt,2*Nh+1))
    dftdxn = np.zeros((Nt,2*Nh+1))
    itn = 0
    while True:
        ftp = ft.copy()
        for l in range(0,Nt):
            if cflg[l]==0: # Separation
                ft[l,:] = 0.0;
                dftdxt[l,:] = 0
                dftdxn[l,0] = 0
                continue
            ft[l,:] = ft[l-1,:] + kt*(xt[l,:]-xt[l-1,:])
            if abs(ft[l,:])>mu*fn[l,:]: # Slip
                ft[l,:] = mu*fn[l,:]*np.sign(ft[l,:])
                dftdxt[l,:] = 0.0
                dftdxn[l,0] = mu*kn*np.sign(ft[l,:])
                for n in range(1,Nh+1):
                    dftdxn[l,2*n-1] = mu*kn*np.cos(n*t[l])*np.sign(ft[l,:])
                    dftdxn[l,2*n] = mu*kn*np.sin(n*t[l])*np.sign(ft[l,:])
            else: # Stick
                dftdxt[l,0] = dftdxt[l-1,0]
                for n in range(1,Nh+1):
                    dftdxt[l,2*n-1] = kt*(np.cos(n*t[l])-np.cos(n*t[l-1])) + dftdxt[l-1,2*n-1]
                    dftdxt[l,2*n] = kt*(np.sin(n*t[l])-np.sin(n*t[l-1])) + dftdxt[l-1,2*n]
                dftdxn[l,:] = dftdxn[l-1,:]
        itn = itn+1
        if la.norm(ftp-ft)<soleps:
            break

    FN = ms.time2freq(fn,Nh)
    FT = ms.time2freq(ft,Nh)
    dFNdXN = ms.time2freq(dfndxn,Nh)
    dFNdXT = np.zeros((2*Nh+1,2*Nh+1))
    dFTdXN = ms.time2freq(dftdxn,Nh)
    dFTdXT = ms.time2freq(dftdxt,Nh)
    return FN,FT,dFNdXN,dFNdXT,dFTdXN,dFTdXT


def eldryfrictsys(X, params):
    Xt = X[:-1,:]
    Xn = np.zeros_like(Xt)
    Xn[0,0] = params['N0']/params['kn']
    FN,FT,dFNdXN,dFNdXT,dFTdXN,dFTdXT = ElDryFrict(Xn,Xt,params)
    dFTdW = np.zeros_like(FT)
    return FT,dFTdXT,dFTdW


Forcinglevels = [ 0.10, 0.50, 1.20, 1.60];

params = dict()
m = 1.0
c = 0.20
k = 1.0
kt = 3.0
kn = 3.0
mu = 1.00
N0 = 1.00
fh = 1
soleps = 1e-10
params['M'] = np.reshape(m,(1,1))
params['C'] = np.reshape(c,(1,1))
params['K'] = np.reshape(k,(1,1))
params['mu'] = mu
params['kt'] = kt
params['kn'] = kn
params['N0'] = N0
params['fh'] = fh
params['eps'] = soleps
params['Fc'] = np.reshape(0.50,(1,1))
params['Fs'] = np.reshape(0.00,(1,1))

Nh = 30;
Nt = 128;
params['Nt'] = Nt
params['Nh'] = Nh

Ws = 0.2
We = 4.00
# Ws = 4.00
# We = 0.1
ds = 0.1
dsmax = 0.1
dsmin = 0.0001
Nop = 6.0
angop = 1.00
E = np.zeros((2*Nh+1,2*Nh+1))
E[0,0] = k+kt
for n in range(1,Nh+1):
    E[2*n-1,2*n-1] = E[2*n,2*n] = k+kt-(n*Ws)**2*m
    E[2*n-1,2*n] = (n*Ws)*c
    E[2*n,2*n-1] = -(n*Ws)*c
Fl = np.zeros((2*Nh+1,1))
Fl[2*fh-1]=1.0
X0 = la.solve(E,Fl)

for F in Forcinglevels:
    params['Fc'] = np.reshape(F,(1,1))
    Xi = ms.hb_freq_cont(eldryfrictsys, X0=X0*F, Ws=Ws, We=We, ds=ds, deriv=True, fnform='Freq', num_harmonics=Nh, eqform='second_order', params=params, num_time_points=Nt, solep=1e-10, ITMAX=100, dsmax=dsmax, dsmin=dsmin, Nop=Nop, angop=angop)
    W = Xi[-1,:]
    A = np.sqrt(Xi[1,:]**2 + Xi[2,:]**2)/np.sqrt(2)
    print(F)
    plt.plot(W,A/F,'o-')
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


# plt.xlim(0.0, 4.5)
# plt.show()
