

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from scipy import interpolate
import cmath
from scipy.optimize import minimize
from scipy.optimize import root

####### Question 1
def K_c (delta, sigma, T):
    a = sigma**2/2*T - sigma*T**0.5*norm.ppf(delta)
    K = 100*np.exp(a)
    return K

def K_p (delta, sigma, T):
    a = sigma**2/2*T + sigma*T**0.5*norm.ppf(delta)
    K = 100*np.exp(a)
    return K

def BS(K, sigma, T):
    d1 = (np.log(100/K)+(sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*100-norm.cdf(d2)*K)
    
def BL_Density (K_list, vol_list, T):
    density = []
    
    for i in range(1,len(K_list)-1):
        c1 = BS(K_list[i-1],vol_list[i-1],T)
        c2 = BS(K_list[i],vol_list[i],T)
        c3 = BS(K_list[i+1],vol_list[i+1],T)
        density += [(c1-2*c2+c3)/0.01]
    
    return density

def density_plot(K, density_1, density_3):
    plt.plot(K[1:-1], density_1)
    plt.plot(K[1:-1], density_3)
    plt.legend(['1M','3M'])
    plt.title("Fig.1 Risk Neutral Density for 1 & 3 Month Options")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.show()

def BL_Density2 (K_list, vol, T):
    density = []
    
    for i in range(1,len(K_list)-1):
        c1 = BS(K_list[i-1],vol,T)
        c2 = BS(K_list[i],vol,T)
        c3 = BS(K_list[i+1],vol,T)
        density += [(c1-2*c2+c3)/0.01]
    
    return density

def density_plot2(K, density_1, density_3):
    plt.plot(K[1:-1], density_1)
    plt.plot(K[1:-1], density_3)
    plt.legend(['1M','3M'])
    plt.title("Fig.2 Risk Neutral Density for 1 & 3 Month Options")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.show()
    
def e1(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*ef1(S[i])*0.1
    
    return price
def ef1(s):
    if s <= 110:
        return 1
    else:
        return 0

def e2(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*ef2(S[i])*0.1
    
    return price

def ef2(s):
    if s >= 105:
        return 1
    else:
        return 0

def e3(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*max(0,S[i]-100)*0.1
    
    return price

####### Question 2
def read_data():
    df = pd.read_excel('mf796-hw5-opt-data.xlsx')
    call = df[['expDays','expT','K','call_bid','call_ask']]
    put = df[['expDays','expT','K','put_bid','put_ask']]
    
    call['mid_price'] = (call['call_ask'] + call['call_bid'])/2
    put['mid_price'] = (put['put_ask'] + put['put_bid'])/2
    call['spread'] = call['call_ask'] - call['call_bid']
    put['spread'] = put['put_ask'] - put['put_bid']
    return call, put

def monotonicity(t,p):
    if t == 'call':
        return all(p == p.cummin())
    else:
        return all(p == p.cummax())

def rate_change(t,p,k):
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
        
def convexity(p):
    n = p-2*p.shift(1) + p.shift(2)
    n.dropna()
    n.index = range(0,len(n))
    a = []
    for i in range(0,len(n)):
        a += n[i]>0
    return all(a)
    
class FFT:
    def __init__(self,lst):
        self.sigma = lst[0]
        self.eta0 = lst[1]
        self.kappa = lst[2]
        self.rho = lst[3]
        self.theta = lst[4]
        self.S0 = 267.15
        self.r = 0.015
        self.q = 0.0177
        self.T = 0.25
        
    def Heston_fft(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        bt = time.time()
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        # Compute FTT
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        # Option price
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        et = time.time()
        
        runt = et-bt

        return(price,runt)
    
    def dirac(self,n):
        """ Define a dirac delta function
        """
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
        
    def Heston_cf(self,u):
        """ Define a function that computes the characteristic function for variance gamma
        """
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        q = self.q
        
        ii = complex(0,1)
        
        l = cmath.sqrt(sigma**2*(u**2+ii*u)+(kappa-ii*rho*sigma*u)**2)
        w = np.exp(ii*u*np.log(S0)+ii*u*(r-q)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(cmath.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*cmath.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        y = w*np.exp(-(u**2+ii*u)*eta0/(l/cmath.tanh(l*T/2)+kappa-ii*rho*sigma*u))
        
        return y  
   
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
            sse += (self.Heston_fft(alpha,n,B,k[i])[0]-c[i])**2
            alpha = -1.5
            sse += (self.Heston_fft(alpha,n,B,k[i])[0]-p[i])**2
    
        return sse
 
    
######## Question 3
        
def BS3(K, S, sigma, r, q, T):
    d1 = (np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-r*T))
    
# In[]    
if __name__ == '__main__':

    # In[]
    ####### Question 2
    call, put = read_data()
    
    print(monotonicity('call', call["mid_price"][0:9]),end=' ')
    print(monotonicity('call', call["mid_price"][9:25]),end=' ')
    print(monotonicity('call', call["mid_price"][25:]),end=' ')
    print(monotonicity('put', put["mid_price"][0:9]),end=' ')
    print(monotonicity('put', put["mid_price"][9:25]),end=' ')
    print(monotonicity('put', put["mid_price"][25:]))
    
    print(rate_change('call', call["mid_price"][0:9],call["K"][0:9]),end=' ')
    print(rate_change('call', call["mid_price"][9:25],call["K"][9:25]),end=' ')
    print(rate_change('call', call["mid_price"][25:],call["K"][25:]),end=' ')
    print(rate_change('put', put["mid_price"][0:9],put["K"][0:9]),end=' ')
    print(rate_change('put', put["mid_price"][9:25],put["K"][9:25]),end=' ')
    print(rate_change('put', put["mid_price"][25:],put["K"][25:]))
    
    print(convexity(call["mid_price"][0:9]),end=' ')
    print(convexity(call["mid_price"][9:25]),end=' ')
    print(convexity(call["mid_price"][25:]),end=' ')
    print(convexity(put["mid_price"][0:9]),end=' ')
    print(convexity(put["mid_price"][9:25]),end=' ')
    print(convexity(put["mid_price"][25:]))
# In[]
    #b
    def funx(x):
        return FFT(x).obj_fxn(opt_data)
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
    x = res.x
    x =(1.23,0.080,0.87,-0.85,0.033) 
    funx(x)
    # In[]
    ####### Question 3
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
    call_heston = FFT(parm).Heston_fft(alpha,n,B,K)[0]
    vol = root(lambda x: BS3(K, S0, x, r, q, expT)-call_heston,0.3).x[0]
    # b
    ds = 0.01
    dv = 0.05 * eta0
    
    parm1 = [sigma,eta0,kappa,rho,theta,S0+ds]
    parm2 = [sigma,eta0,kappa,rho,theta,S0-ds]
    parm3 = [sigma,eta0+dv,kappa,rho,theta+dv,S0]
    parm4 = [sigma,eta0-dv,kappa,rho,theta-dv,S0]
    delta_heston = (FFT(parm1).Heston_fft(alpha,n,B,K)[0]-FFT(parm2).Heston_fft(alpha,n,B,K)[0])/(2*ds)
    vega_heston = (FFT(parm3).Heston_fft(alpha,n,B,K)[0]-FFT(parm4).Heston_fft(alpha,n,B,K)[0])/(2*dv)
    
    d1 = (np.log(S0/K)+(r-q+vol**2/2)*expT)/(vol*(expT)**0.5)
    delta_bs = np.exp(-q*expT)*norm.cdf(d1)
    vega_bs = np.exp(-q*expT)*S0*np.sqrt(expT)*norm.cdf(d1)
    delta_bs
    delta_heston
    vega_heston
    vega_bs
    
    
    # c
    K = S0
    straddle_p = 2 * FFT(parm3).Heston_fft(alpha,n,B,K)[0] + K*np.exp(-r*expT)-(S0)*np.exp(-q*expT)
    straddle_m = 2 * FFT(parm4).Heston_fft(alpha,n,B,K)[0] + K*np.exp(-r*expT)-(S0)*np.exp(-q*expT)
    vega_heston_straddle = (straddle_p-straddle_m)/(2*dv)
    vega_heston_straddle
    
    
    
    
    
    
    
    
    