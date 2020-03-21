# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:52:55 2020

@author: Zackt
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import cmath
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import root
import time
class q1:
    def __init__(self, stock, risk_free):
        self.p = stock
        self.r = risk_free
        
    def Kc (self, delta , t, sigma):
        p = self.p
        k = p * np.exp(sigma**2 / 2 * t - sigma * np.sqrt(t) * norm.ppf(delta))

        return k
    
    def Kp (self, delta, t, sigma):
        p = self.p
        k = p * np.exp(sigma**2 / 2 * t + sigma * np.sqrt(t) * norm.ppf(delta))
        return k 
    
    def bsmodel(self,k, sigma , T):
        s = self.p
        d1 = (np.log( s / k )+( sigma ** 2 / 2 ) * T)/( sigma * T ** 0.5)
        d2 = d1 - sigma * T **0.5
        p = (norm.cdf(d1)* s -norm.cdf(d2)*k)
        return p

    def BL_Density (self,K_list, vol_list, T):
        density = []
        
        for i in range(1,len(K_list)-1):
            c1 = self.bsmodel(K_list[i-1],vol_list[i-1],T)
            c2 = self.bsmodel(K_list[i],vol_list[i],T)
            c3 = self.bsmodel(K_list[i+1],vol_list[i+1],T)
            density += [(c1-2*c2+c3)/0.01]
    
        return density

    def density_plot(self, K, density_1, density_3):
        plt.plot(K[1:-1], density_1 , label = '1m')
        plt.plot(K[1:-1], density_3 , label = '3m')
        plt.legend()
        plt.title("Risk Neutral Density")
        plt.xlabel("strike price")
        plt.ylabel("Density")
        plt.show()
           

    def e(self,density1, density2,density3 ,s):
        price1 = 0
        for i in range(0,len(s)-2):
            if s[i] <= 110:
                ss = 1
            else:
                ss = 0
            price1 += density1[i]*ss*0.1
            
            
        price2 = 0
        for i in range(0,len(s)-2):
            ss = 0
            if s[i] >= 105:
                ss = 1
            else:
                ss = 0
            price2 += density2[i]*ss*0.1        
            
            
        price3 = 0
        for i in range(0,len(s)-2):
            price3 += density3[i]*max(0,s[i]-100)*0.1
    
        return price1 , price2, price3
 

class Hw3_FFT:
    def __init__(self, x):
        self.sigma = x[0]
        self.v0 = x[1]
        self.kappa = x[2]
        self.p = x[3]
        self.theta = x[4]
        self.s0 = 267.15
        self.r =0.015
        self.k = 0.0177
        self.T = 0.25        
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
    
    def Heston_Model_Hw3_FFT(self, alpha,n,B,K):
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
    def obj_fxn(self,data):
        self.S0 = data['s0'][0]
        self.r = data['r'][0]
        self.q = data['q'][0]
        self.T = data['expDays_call'][0]/365
        sse = 0
        c = data["mid_price_call"][0:9]
        p = data["mid_price_put"][0:9]
        k = data["K_call"][0:9]
        k.index = range(0,len(k))
       
        for i in range(len(k)):
            alpha = 1
            n = 12
            B = 1000
            sse += (self.Heston_Model_Hw3_FFT(alpha,n,B,k[i])[0]-c[i])**2
            alpha = -1.5
            sse += (self.Heston_Model_Hw3_FFT(alpha,n,B,k[i])[0]-p[i])**2
    
        return sse    
class q2:
    def __init__(self, df):
        self.df = df
        
    def test_arbitrage(self):
        df = self.df
        call = df[['expDays','expT','K','call_bid','call_ask']]
        put = df[['expDays','expT','K','put_bid','put_ask']]
        
        call['mid_price'] = (call['call_ask'] + call['call_bid'])/2
        put['mid_price'] = (put['put_ask'] + put['put_bid'])/2
        call['spread'] = call['call_ask'] - call['call_bid']
        put['spread'] = put['put_ask'] - put['put_bid']
        return call, put
    def monotonicity(self,t,p):
        if t == 'call':
            return all(p == p.cummin())
        else:
            return all(p == p.cummax())
    
    def rate_change(self,t,p,k):
        if t == 'call':
            r = (p.shift(1)-p)/(k.shift(1)-k)
            r = r.dropna()
            r.index = range(0,len(r))
            a = []
            for i in range(0,len(r)):
                a += r[i]>-1 and r[i] < 0
            return all(a)
        else:
            r = (p.shift(1)-p)/(k.shift(1)-k)
            r = r.dropna()
            r.index = range(0,len(r))
            a = []
            for i in range(0,len(r)):
                a += r[i]>0 and r[i] < 1
            return all(a)
            
    def convexity(self,p):
        n = p-2*p.shift(1) + p.shift(2)
        n.dropna()
        n.index = range(0,len(n))
        a = []
        for i in range(0,len(n)):
            a += n[i]>0
        return all(a)
# In[]
table = np.matrix([[0.1,0.25,0.4,0.5,0.4,0.25,0.1],\
                   [0.3225, 0.2473, 0.2021, 0.1824, 0.1574, 0.1370, 0.1148],\
                   [0.2836, 0.2178,0.18158, 0.1645, 0.1462, 0.1256, 0.1094]]).T
question1 = q1(100,0)
m1 = [question1.Kp(table[i,0] , 1/12,table[i,1] )if i < 4 else question1.Kc(table[i,0] , 1/12,table[i,1] ) for i in range(table.shape[0])]
m3 = [question1.Kp(table[i,0] , 3/12,table[i,2] )if i < 4 else question1.Kc(table[i,0] , 3/12,table[i,2] ) for i in range(table.shape[0])]
df = pd.DataFrame({'m1':m1 , 'm3' :m3} , index = table[:,0])
# In[]
print(np.polyfit(m1,table[:,1],1))
print(np.polyfit(m3,table[:,2],1))
# In[]
K_list = np.linspace(75,112.5,375)
vol_1 = 1.5578 - 0.0138*K_list
vol_3 = 0.9324 - 0.0077*K_list
density_1 = np.array(question1.BL_Density(K_list, vol_1, 1/12))
density_3 = np.array(question1.BL_Density(K_list, vol_3, 3/12))
question1.density_plot(K_list, density_1, density_3)

# In[]
K_list = np.linspace(75,112.5,375)
vol_1 = [0.1824] * 375
vol_3 = [0.1645] * 375
density_1 = np.array(question1.BL_Density(K_list, vol_1, 1/12))
density_3 = np.array(question1.BL_Density(K_list, vol_3, 3/12))
question1.density_plot(K_list, density_1, density_3)

# In[e]
vol_1 = 1.5578 - 0.0138*K_list
vol_3 = 0.9324 - 0.0077*K_list
vol_2 = (vol_1+vol_3)/2
density_2 = np.array(question1.BL_Density(K_list, vol_2, 2/12))
S = np.linspace(75,112.5,375)
question1.e(density_1,density_3,density_2,S)
# In[]
'''----------------------------b----------------------------'''
df = pd.read_excel('D:/HW/Homework 5 - Breeden-Litzenberger/mf796-hw5-opt-data.xlsx')
q = q2(df)
call, put = q.test_arbitrage()
print(q.monotonicity('call', call["mid_price"][0:9]),end=' ')
print(q.monotonicity('call', call["mid_price"][9:25]),end=' ')
print(q.monotonicity('call', call["mid_price"][25:]),end=' ')
print(q.monotonicity('put', put["mid_price"][0:9]),end=' ')
print(q.monotonicity('put', put["mid_price"][9:25]),end=' ')
print(q.monotonicity('put', put["mid_price"][25:]))

print(q.rate_change('call', call["mid_price"][0:9],call["K"][0:9]),end=' ')
print(q.rate_change('call', call["mid_price"][9:25],call["K"][9:25]),end=' ')
print(q.rate_change('call', call["mid_price"][25:],call["K"][25:]),end=' ')
print(q.rate_change('put', put["mid_price"][0:9],put["K"][0:9]),end=' ')
print(q.rate_change('put', put["mid_price"][9:25],put["K"][9:25]),end=' ')
print(q.rate_change('put', put["mid_price"][25:],put["K"][25:]))

print(q.convexity(call["mid_price"][0:9]),end=' ')
print(q.convexity(call["mid_price"][9:25]),end=' ')
print(q.convexity(call["mid_price"][25:]),end=' ')
print(q.convexity(put["mid_price"][0:9]),end=' ')
print(q.convexity(put["mid_price"][9:25]),end=' ')
print(q.convexity(put["mid_price"][25:]))
# In[]
def funx(x):
    return Hw3_FFT(x).obj_fxn(opt_data)
r = 0.015
q = 0.0177
s0 = 267.15
opt_data = call
opt_data = opt_data.join(put, lsuffix='_call', rsuffix='_put')
opt_data['r'] = r
opt_data['q'] = q
opt_data['s0'] = s0
guess = [1.5,0.1,0.25,-0.5,0.1]
lower = [0.001,  0.0, 0.0, -1, 0.0]
upper = [5.0,  2.0, 2.0, 1, 1.0]
res = minimize(funx,guess,method='nelder-mead')

# In[]
        
def BS3(K, S, sigma, r, q, T):
    d1 = (np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-r*T))
kappa = 3.51
theta = 0.052
sigma = 1.17
rho = -0.77
eta0 = 0.034
S0 = 267.15
r = 0.015
q = 0.0177
expT = 0.25
K = 275
alpha = 1
n = 15
B = 1000
parm = [sigma,eta0,kappa,rho,theta,S0]
call_heston = Hw3_FFT(parm).Heston_Model_Hw3_FFT(alpha,n,B,K)[0]
vol = root(lambda x: BS3(K, S0, x, r, q, expT)-call_heston,0.3).x[0]

# In[]
ds = 0.01
dv = 0.05 * eta0

parm1 = [sigma,eta0,kappa,rho,theta,S0+ds]
parm2 = [sigma,eta0,kappa,rho,theta,S0-ds]
parm3 = [sigma,eta0+dv,kappa,rho,theta+dv,S0]
parm4 = [sigma,eta0-dv,kappa,rho,theta-dv,S0]
delta_heston = (Hw3_FFT(parm1).Heston_Model_Hw3_FFT(alpha,n,B,K)[0]-Hw3_FFT(parm2).Heston_Model_Hw3_FFT(alpha,n,B,K)[0])/(2*ds)
vega_heston = (Hw3_FFT(parm3).Heston_Model_Hw3_FFT(alpha,n,B,K)[0]-Hw3_FFT(parm4).Heston_Model_Hw3_FFT(alpha,n,B,K)[0])/(2*dv)
d1 = (np.log(S0/K)+(r-q+vol**2/2)*expT)/(vol*(expT)**0.5)
delta_bs = np.exp(-q*expT)*norm.cdf(d1)
vega_bs = np.exp(-q*expT)*S0*np.sqrt(expT)*norm.cdf(d1)   