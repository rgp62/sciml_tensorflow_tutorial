import tensorflow as tf
import numpy as np

def makeNN(width,depth,dimi,dimo):
    return tf.keras.models.Sequential(
                             [tf.keras.layers.Dense(width,activation='elu',dtype=tf.float64) for _ in range(depth-2)]
                            +[tf.keras.layers.Dense(dimo,dtype=tf.float64)]
                         )        

class DeepONetUnstacked(tf.keras.Model):
    def __init__(self,width,depth,dimu,dimy):
        super().__init__()
        self.b = makeNN(width,depth,dimu,width)
        self.t = makeNN(width,depth,dimy,width)

    @tf.function
    def __call__(self,u,y):
        return tf.einsum('ab,cb->ac',self.b(u),self.t(y))

class MORP(tf.keras.Model):
    def __init__(self,depth,width,k):
        super().__init__()
        self.h = makeNN(depth,width,1,1)
        self.g = makeNN(depth,width,1,2)
        self.k = k
    @tf.function
    def __call__(self,u):
        gk = self.g(self.k[:,None])
        gk = tf.complex(gk[...,0],gk[...,1])
        hu = self.h(u[...,None])[...,0]
        huh = tf.signal.rfft(hu)
        Luh = gk * huh
        Lu = tf.signal.irfft(Luh)
        return Lu
    
class SumOPs(tf.keras.Model):
    def __init__(self,ops):
        super().__init__()
        self.ops = ops
        
    @tf.function
    def __call__(self,u):
        return sum([op(u) for op in self.ops])
        
class FwdEuler(tf.keras.Model):
    def __init__(self,op,dt):
        super().__init__()
        self.op = op
        self.dt = dt
        
    @tf.function
    def calli(self,a,t):
        return a+self.dt*self.op(a)
    
    @tf.function
    def callNt(self,u0,Nt):
        return tf.foldl(self.calli,np.arange(Nt),initializer=u0)
        
    @tf.function
    def __call__(self,u0,Nt):
        return tf.transpose(tf.scan(self.calli,np.arange(Nt),initializer=u0),(1,0,2))


    
    





