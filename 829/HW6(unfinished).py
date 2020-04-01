#!/usr/bin/env python
# coding: utf-8

# In[28]:


from scipy.stats import norm
import numpy as np
sigma = 22.5/100
T = 3/12
t = 0
s = 1200
k = 1200
r = np.log(100/98.45) /3 * 12

def bs(s,k,r,T,t,sigma):
    d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) - (r + sigma**2 /2)*(T - t))
    d2 = d1 - (sigma * (T - t)**0.5)
    return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(r * (T - t))
bs(s,k,r,T,t,sigma)


# In[ ]:




