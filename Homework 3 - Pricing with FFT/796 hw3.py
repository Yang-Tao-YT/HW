# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:59:29 2020

@author: Yang
"""

# In[]
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import cmath
from scipy import interpolate
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pylab import meshgrid
import time
from scipy.optimize import root
import  scipy.stats as  sc
class Hw3_bs:
    def __init__(self, s = 10, k = 12, t = 3/12 , rate = 0.04 , vol = 0.2 ):
        self.s = s
        self.k = k
        self.t = t
        self.r = rate
        self.sigma = vol
        
    def euro_buy(self):

    
        self.d1 = (np.log(self.s/self.k) + self.t*(self.r+(self.sigma**2)/2)) / np.sqrt(self.sigma**2 * self.t)
        self.d2 = (np.log(self.s/self.k) + self.t*(self.r-(self.sigma**2)/2)) / np.sqrt(self.sigma**2 * self.t)
    
        self.buy= (self.s * norm.cdf(self.d1) - self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2) )
        return self.buy    

class Hw3_FFT:
    def __init__(self, sigma, v0, kappa, p, theta, s0, r, k,T):
        self.sigma = sigma
        self.v0 = v0
        self.kappa = kappa
        self.p = p
        self.theta = theta
        self.s0 = s0
        self.r =r
        self.k = k
        self.T = T
        
    def  Heston_Model_CF (self, u):
        '''charactristic function of density'''
        theta = self.theta
        kappa = self.kappa
        p = self.p
        sigma =self.sigma
        t = self.T
        s0 = self.s0
        r = self.r
        i = complex(0,1)
        # lambda
        Lambda = np.sqrt(sigma**2 * (u**2 + i * u) + (kappa - i * p * sigma * u)**2)
        # w
        wu = np.exp(i * u * np.log(s0) + i * u * (r - 0) * t  + \
            (kappa * theta * t * (kappa - i * p * sigma * u)) / sigma**2)

        wd = (cmath.cosh(Lambda * t / 2) + (kappa - i * p * sigma * u)/
                   Lambda * cmath.sinh(Lambda * t / 2))**(2 * kappa * theta / sigma**2)
        w = wu/wd
        # cf
        cf = w * np.exp( -(u**2 + i * u) * self.v0/
                (Lambda / cmath.tanh(Lambda * t / 2) + kappa - i * p * sigma * u))
        return cf
    
    def indicator (self,n):
        '''indicator factor'''
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
    
    def Heston_Model_FFT(self, alpha,n,B,K):
        '''calculate the price via FFT'''
        start = time.time()
                # set the parameter
        a = alpha
        theta = self.theta
        kappa = self.kappa
        p = self.p
        sigma =self.sigma
        t = self.T
        s0 = self.s0
        r = self.r
        N = 2**n
        dv = B / N
        dvdk = 2 * np.pi / N
        dk = dvdk / dv
        j = np.arange(1,N+1,dtype = complex)
        vj = (j - 1) * dv
        m = np.arange(1,N+1,dtype = complex)
        beta = np.log(s0) - dk * N / 2
        km = beta + (m-1) *dk
        i = complex(0,1)
        vj_ = []
        
        
#       set x and calculate y
        for count in range(0,N):
            u = vj[count] - (a + 1) * i
            cf = self.Heston_Model_CF(u)
            res =cf / ( 2 * (a +  vj[count] * i ) * (a + vj[count] * i + 1)) 
            vj_.append(res)
        vj_ = np.array(vj_)
        x = dv *vj_  * np.exp(- i * vj * beta) * (2 - self.indicator(j-1)) 
        y = np.fft.fft(x)
        
        
        # vector 
        yreal = np.exp(-alpha * np.array(km)) / np.pi * np.array(y).real
        
        # k_list
        k_List = list(beta + (np.cumsum(np.ones((N, 1))) - 1) * dk)
        kt = np.exp(np.array(k_List))
        k = []
        yreal_check = []
        #make sure the data is valid
        for i in range(len(kt)):
            if( kt[i]>1e-16 )&(kt[i] < 1e16)& ( kt[i] != float("inf"))&( kt[i] != float("-inf")) &( yreal[i] != float("inf"))&(yreal[i] != float("-inf")) & (yreal[i] is not  float("nan")):
                k.append(kt[i])
                yreal_check.append(yreal[i])
        tck = interpolate.splrep(k , np.real(yreal_check))
        price =  np.exp(-r * t)*interpolate.splev(K, tck).real
        end = time.time()
        runtime = end - start
        return price,runtime
    
    def plot(self,ns , Bs , k , standard):


        x,y = meshgrid(Bs, ns)
        z = np.zeros([len(ns), len(Bs)])
        times = np.zeros([len(ns), len(Bs)])
        diff = np.zeros([len(ns), len(Bs)])
        x, y = np.meshgrid(ns, Bs)
        efficient = []
        for n in range(len(ns)):
            for B in range(len(Bs)):
                r = Hw.Heston_Model_FFT(1,ns[n],Bs[B], k)
                z[n][B] = r[0]
                times [n][B] = r[1]
                diff[n][B] = 1/ ((r[0] - standard)**2*times[n][B]) 
                efficient.append([diff[n][B] ,ns[n] , Bs[B] ])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z.T, rstride=1, cstride=1)
        ax.set_title('price vs n vs B', fontsize = 30)
        ax.set_xlabel("N", fontsize = 30)
        ax.set_ylabel("B", fontsize = 30)
        ax.set_zlabel("price", fontsize = 30)
        plt.show()
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, diff.T, rstride=1, cstride=1)
        ax.set_title('efficiency vs n vs B', fontsize = 30)
        ax.set_xlabel("N", fontsize = 30)
        ax.set_ylabel("B", fontsize = 30)
        ax.set_zlabel("efficiency", fontsize = 30)
        plt.show()
        return (z , diff , times , max(efficient))
    
    def vol_vs_K(self,price_list,K_list):
        r = self.r
        t = self.T
        s = self.s0
        vol = []
        for i in range(len(K_list)):


            result = root(lambda x: BS(s , K_list[i], t, r,sigma =x).BSeurocall()-price_list[i],0.3)
            vol += [result.x]
        
        vol = np.array(vol)
        plt.plot(K_list,vol)
        plt.title("V vs K", fontsize = 30)
        plt.xlabel("K", fontsize = 30)
        plt.ylabel("V", fontsize = 30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.show()
        
    def vol_vs_T(self,price_list,t_list):
        vol = []
        r = self.r
        k = self.k
        s = self.s0        
        for i in range(len(t_list)):
            result = root(lambda x:BS(s , k, t_list[i], r,sigma =x).BSeurocall()-price_list[i],0.3)
            vol += [result.x]
            
        vol = np.array(vol)
        plt.plot(t_list,vol)
        plt.title("V vs T", fontsize = 30)
        plt.xlabel("T", fontsize = 30)
        plt.ylabel("V", fontsize = 30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.show()
class BS:
    def __init__(self,s,k,t,r,sigma):
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        
    def BSeurocall(self):
        
        d1 = (np.log(self.s / self.k ) + (self.r + self.sigma ** 2 / 2) * self.t) / (
                self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        
        Nd1 = sc.norm.cdf(d1)
        Nd2 = sc.norm.cdf(d2)
        
        eurocall = Nd1 * self.s - Nd2 * self.k * np.exp( -1 * self.r * self.t)
        
        return eurocall
    
# In[]
if __name__ == '__main__':
    '''initial class for hw 1'''
    Hw = Hw3_FFT(sigma = 0.2, v0 = 0.08,kappa= 0.7,p= -0.4,theta= 0.1,s0= 250,r= 0.02,k= 250,T= 0.5)
    s = Hw.Heston_Model_FFT(1.5,11, 675 , 250)
    # In[]
    '''1'''
    alpha = np.arange(0.1,35,0.5)
    test = []
    for i in alpha:
        test += [Hw.Heston_Model_FFT(i , 11,250*2.7 , 250)[0]]
    # In[]
    '''plot test'''
    plt.figure(figsize = [20,15])
    plt.plot(alpha,test , linewidth = 4) 
    plt.xlabel('Alpha' , fontsize = 30)
    plt.ylabel('option price', fontsize = 30)
    plt.title('Alpha vs option price - stable', fontsize = 30)
    plt.xticks([alpha[i] - 0.1 for i in range(len(alpha)) if i%10 == 0] , fontsize = 30)
    plt.yticks(fontsize =30)
    plt.show()
    # In[]
    '''table '''
    alpha = list(np.arange(0.0000000000001,40,0.5))
    test = []
    for i in alpha:
        test += [Hw.Heston_Model_FFT(i , 11,250*2.7 , 250)]
        # In[]
    alpha = alpha[:5] + alpha[75:]
    test = test [:5] + test [75:]
    df = pd.DataFrame({'alpha' : alpha , 'test' : test}).T
# In[]
    '''compare the different choice of B and N'''
    Bs = np.linspace(250*2.5,250*2.7,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    mean,diff,times, maximun = Hw.plot(ns,Bs,250 , 21.27 )

 # In[]
    '''change k to the 260'''
    Hw = Hw3_FFT(sigma = 0.2, v0 = 0.08,kappa= 0.7,p= -0.4,theta= 0.1,s0= 250,r= 0.02,k= 260,T= 0.5)

    s = Hw.Heston_Model_FFT(1.5,11, 675 , 260)[0]

    
    # In[]
    '''compare the different choice of B and N to find the best n and k'''
    Bs = np.linspace(250*2.5,250*2.7,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    mean,diff,times, maximun = Hw.plot(ns,Bs,260 ,  s  )


# In[]
    #-----------------------------b------------------------
    Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= 0.5,p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= 0.25)
    s = Hw.Heston_Model_FFT(1.5,9, 150*2.7 , 260)[0]
    K = np.linspace(80,230,60)
    price = []
    for i in K:
        price.append(Hw.Heston_Model_FFT(1.5,9, 150*2.7 , i)[0])
    plt.plot(K,price)
    plt.title("Call Option Price vs K")
    plt.xlabel("K")
    plt.ylabel("Call Option Price")
    plt.show()        
    Hw.vol_vs_K(price,K)
# In[]
    '''t'''
    t_list= np.linspace(1/12,2,50)
    price = []
    for t in t_list:
        Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= 0.5,\
              p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= t)
        price += [Hw.Heston_Model_FFT(1.5,9,150*2.7,150)[0]]

    Hw.vol_vs_T(price,t_list)    
    price = np.array(price)
    plt.plot(t_list,price)
    plt.title("Call Option Price vs T")
    plt.xlabel("T")
    plt.ylabel("Call Option Price")
    plt.show()
    Hw.vol_vs_T(price,t_list)
# In[]
    #---------------------3----------------------
    #------------------v0------------------------
    '''v0 = 0.3'''
    v0 = 0.3
    Hw = Hw3_FFT(sigma = 0.4, v0 = v0,kappa= 0.5,p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= 0.25)
    s = Hw.Heston_Model_FFT(1.5,9, 150*2.7 , 260)[0]
    K = np.linspace(80,230,60)
    price = []
    for i in K:
        price.append(Hw.Heston_Model_FFT(1.5,9, 150*2.7 , i)[0])
       
    Hw.vol_vs_K(price,K)
    
    '''t'''
    t_list= np.linspace(1/12,2,50)
    price = []
    for t in t_list:
        Hw = Hw3_FFT(sigma = 0.4, v0 = v0,kappa= 0.5,\
              p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= t)
        price += [Hw.Heston_Model_FFT(1.5,9,150*2.7,150)[0]]

    Hw.vol_vs_T(price,t_list)
# In[]
    #-------------- kappa---------------------
    '''kappa = 0'''
    kappa = 0
    Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= 0.25)
    s = Hw.Heston_Model_FFT(1.5,9, 150*2.7 , 260)[0]
    K = np.linspace(80,230,60)
    price = []
    for i in K:
        price.append(Hw.Heston_Model_FFT(1.5,9, 150*2.7 , i)[0])
       
    Hw.vol_vs_K(price,K)
    
    '''t'''
    t_list= np.linspace(1/12,2,50)
    price = []
    for t in t_list:
        Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,\
              p= 0.25,theta= 0.12,s0= 150,r= 0.025,k= 150,T= t)
        price += [Hw.Heston_Model_FFT(1.5,9,150*2.7,150)[0]]

    Hw.vol_vs_T(price,t_list)

# In[]
    #---------------------p--------------
    '''rho = -0.25 '''
    kappa = 0.5
    p = -0.25
    Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,p=p,theta= 0.12,s0= 150,r= 0.025,k= 150,T= 0.25)
    s = Hw.Heston_Model_FFT(1.5,9, 150*2.7 , 260)[0]
    K = np.linspace(80,230,60)
    price = []
    for i in K:
        price.append(Hw.Heston_Model_FFT(1.5,9, 150*2.7 , i)[0])
       
    Hw.vol_vs_K(price,K)
    
    '''t'''
    t_list= np.linspace(1/12,2,50)
    price = []
    for t in t_list:
        Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,\
              p= p,theta= 0.12,s0= 150,r= 0.025,k= 150,T= t)
        price += [Hw.Heston_Model_FFT(1.5,9,150*2.7,150)[0]]

    Hw.vol_vs_T(price,t_list)
# In[]
    #-----------------------------theta---------------
    '''theat = 0.3'''
    kappa = 0.5
    theta = 0.3
    Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,p= 0.25,theta= theta,s0= 150,r= 0.025,k= 150,T= 0.25)
    s = Hw.Heston_Model_FFT(1.5,9, 150*2.7 , 260)[0]
    K = np.linspace(80,230,60)
    price = []
    for i in K:
        price.append(Hw.Heston_Model_FFT(1.5,9, 150*2.7 , i)[0])
       
    Hw.vol_vs_K(price,K)
    
    '''t'''
    t_list= np.linspace(1/12,2,50)
    price = []
    for t in t_list:
        Hw = Hw3_FFT(sigma = 0.4, v0 = 0.09,kappa= kappa,\
              p= 0.25,theta= theta,s0= 150,r= 0.025,k= 150,T= t)
        price += [Hw.Heston_Model_FFT(1.5,9,150*2.7,150)[0]]

    Hw.vol_vs_T(price,t_list)
# In[]
    #-------------------