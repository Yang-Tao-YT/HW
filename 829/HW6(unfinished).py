#!/usr/bin/env python
# coding: utf-8

# In[174]:


from scipy.stats import norm
import numpy as np
sigma = 22.5/100
T = 3/12
t = 0
s = 1200
k = 1200
r = np.log(100/98.45) /3 * 12
q = 2 /100
def bs(s,k,r,T,t,sigma,q):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r - q + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    print(d1,d2)
    return -norm.cdf(-d1) * s * np.exp(-q * (T - t)) + norm.cdf(-d2) * k * np.exp(-r * (T - t))
bs(s,k,r,T,t,sigma,q)


# #----------question 2

# In[172]:


sigma = 23.5/100
T = 3/12
t = 0
s = 49.5
k = 50
r = np.log(100/98.55) /3 * 12
s = s - 1/(100/ 99.53 ) * 0.75
def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-r * (T - t))
print(bs(s,k,r,T,t,sigma))

T = 1/12
k = 50 - 0.75
def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-r * (T - t))
print(bs(s,k,r,T,t,sigma))


# #----------question 3 currency option - call

# In[21]:


sigma = 15/100
T = 3/12
t = 0
s =  108
k =  110
r = np.log(100/98.45) /3 * 12
rf = np.log(1000/993) /3 * 12
def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r - rf + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s * np.exp(-rf * (T - t)) - norm.cdf(d2) * k * np.exp(-r * (T - t))


# In[22]:


bs(s,k,r,T,t,sigma)


# question 4

# In[154]:


from scipy.optimize import fsolve
def fun(d):
    sigma = 30/100
    T = 6/12
    t = 0
    s =  18
    k =  20
    r = 10/100
    s = s - np.exp(- r * (2/12)) * d
    mat = bs(s,k,r,T,t,sigma)
    k = k -d
    T = 2/12
    early = bs(s,k,r,T,t,sigma)
    result = mat - early
    return result
        

print(fsolve(fun,0.3))


# In[156]:


def fun(d):
    sigma = 30/100
    T = 6/12
    t = 0
    s =  18
    k =  20
    r = 10/100
    s = s - np.exp(- r * (2/12)) * d  - np.exp( - r * (5/12))*d
    mat = bs(s,k,r,T,t,sigma)
    k = k -d - d
    T = 5/12
    early = bs(s,k,r,T,t,sigma)
    result = mat - early
    return result
print(fsolve(fun,0))


# In[173]:


sigma = 30/100
T = 6/12
t = 0
s =  18
k =  20
r = 10/100
d = 0.4
s = s - np.exp(- r * (2/12)) * d
def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-r * (T - t))
print(bs(s,k,r,T,t,sigma))

T = 2/12
k = k -d

print(bs(s,k,r,T,t,sigma))


# In[170]:


sigma = 30/100
T = 6/12
t = 0
s =  18
k =  20
r = 10/100
d = 0.4
s = s - np.exp(- r * (2/12)) * d  - np.exp(- r * (5/12)) * d 
def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-r * (T - t))
print(bs(s,k,r,T,t,sigma))

k = k - d - d
T = 5/12

def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) + (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-r * (T - t))
print(bs(s,k,r,T,t,sigma))


# In[ ]:




