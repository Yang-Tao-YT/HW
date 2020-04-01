#!/usr/bin/env python
# coding: utf-8

# In[40]:


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
    return -norm.cdf(-d1) * s + norm.cdf(-d2) * k * np.exp(-r * (T - t))
bs(s,k,r,T,t,sigma,q)


# In[ ]:





# In[ ]:




