import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow_probability as tfp
import sympy
import scipy

def get_ex5():
    x_data = np.linspace(-1,1,100)
    x = sympy.Symbol('x')
    u = x*(1-x)
    k = .25*sympy.sin(sympy.pi*x)+.5
    Lu = sympy.diff(k*sympy.diff(u))
    fu = sympy.lambdify([x],u)
    fLu = sympy.lambdify([x],Lu)
    fk = sympy.lambdify([x],k)
    return x_data, fu(x_data), fLu(x_data), fk(x_data)

def IC(samps,k):
    uh = 20*np.exp(2*np.pi*np.random.normal(0,1,(samps,len(k)))*1.j)*scipy.special.erfc(k-3)
    u = np.fft.irfft(uh)
    return u
def upd(u,dt,k):
    return u+dt*(-np.fft.irfft(1.j*k*np.fft.rfft(u**2) + 1e-1*k**2*np.fft.rfft(u)))
def burgersEqn(samps,dt,Nt,k):
    u = [IC(samps,k)]
    for t in range(Nt):
        u.append(upd(u[-1],dt,k))
    return np.stack(u,1)

def get_ex3_1():
    N = 256
    samps = 1
    L = 2.*np.pi
    x = np.arange(0,N)/N*L
    k = np.arange(0,N//2+1)
    T = 1.
    Nt = 1000
    dt = T/Nt
    t = np.arange(Nt+1)*dt
    u = burgersEqn(1,dt,Nt,k)
    return x,t,u[0]

def get_ex3_2():
    xt = np.random.uniform(-1,1,(1000,2)).astype('float32')
    return xt,-1/9 * (tf.sin(3*xt[...,0]) + tf.sin(3*xt[...,1]))

def get_ex6_1(samps):
    N = 256
    L = np.pi
    x = np.arange(0,N)/N*L
    k = np.arange(0,N//2+1)    
    uh = 20*np.exp(2*np.pi*np.random.normal(0,1,(samps,N//2+1))*1.j)*scipy.special.erfc(k-3)
    u = np.fft.irfft(uh)
    v = np.fft.irfft(1.j*k*np.fft.rfft(u**2))
    return u,v

def get_ex6_2(samps):
    N = 256
    L = 2.*np.pi
    x = np.arange(0,N)/N*L
    k = np.arange(0,N//2+1)
    T = 1.
    Nt = 1000
    dt = T/Nt
    t = np.arange(Nt+1)*dt
    u = burgersEqn(1,dt,Nt,k)
    return x,t,u,k

