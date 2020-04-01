#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import norm
import numpy as np
sigma = 22.5/100
T = 3/12
t = 0
s = 1200
k = 1200
r = (100/98.45 - 1) /3 * 12


# In[2]:


d1 = 1/(sigma * (T - t)**0.5) * (np.log(s/k) - (r + sigma**2 /2)*(T - t))


# In[3]:


d2 = d1 - (sigma * (T - t)**0.5)


# In[5]:


norm.cdf(d1) * s - norm.cdf(d2) * k * 1/(100/98.45)


# In[ ]:




