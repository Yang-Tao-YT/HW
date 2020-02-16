# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:13:49 2020

@author: Zackt
"""
# In[]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

class hw4_eigenvector:
    def __init__(self, x , period , data = None):
        if data == 'df': # read the preproccesd data
            self.df = x
        else:
            dic = {i : yf.Ticker(i).history(period = period)['Close'] \
                   for i in x}
            self.df = pd.DataFrame(dic)
            
    def _clean(self):
        '''clean the data'''
        self.df = self.df.fillna(method = 'ffill').dropna()
    
    def log_return(self):
        '''calculate the log return'''
        df = self.df
        t_0 = df.shift(1) #t-1
        ratio = df / t_0 # t/t-1
        log_return = np.log(ratio)
        return log_return.dropna()
    
    def covariance_matrix(self):
        log_return = self.log_return()
        cm = log_return.cov()
        return cm
        
    def find_cover(self,percentage):
        '''find the vector that account for certain percentage of variance'''
        cov = self.covariance_matrix()
        value , vector =  np.linalg.eig(cov)
        value = value / np.sum(value)
        dic = {value[i] : vector[:,i] for i in range(len(value))}
        value = np.sort(value)[::-1]
        cumsum = np.cumsum(value)
        count = [value[i] for i in range(len(value)) if cumsum[i] < percentage]
        count += [value[len(count)+1]]
        member = {count[i] : dic[count[i]] for i in range(len(count))}
        return  member 
    
    
    #----------------------------question2------------------------
    def svd (self):
        '''Singular Value Decomposition'''
        c = self.covariance_matrix()
        U,D,VT = np.linalg.svd(c)
        D = np.diag(D)
        return U,D,VT
    
    
    def GCG(self , G):

        '''find the GCG with svd and pseudo inverse'''
        u , d, vt = self.svd()
        c_inverse = np.dot(np.dot(u , np.linalg.inv(d)) , vt)
        
        gcg = np.dot(np.dot(G , c_inverse) , G.T)
        return gcg,c_inverse
    
    def portfolio(self, G,a,constant):
        cons = np.array([1,0.1])
        gcg, c_inverse = self.GCG(G)
        gcg_inverse = np.linalg.inv(gcg)
        mean = self.log_return().mean()
        lamda =gcg_inverse @ (G @ c_inverse @ mean - 2 * a * cons)
        weight  = 1 /(2 * a) * c_inverse @ (mean - G.T @ lamda)
        return weight,c_inverse
# In[]
if __name__ == '__main__':
    #tick = pd.read_csv('C:/Users/Zackt/Documents/1233.csv')
    #ticker = list(df.columns)
    df = pd.read_csv('C:/Users/Zackt/Documents/100 stock price 5 years.csv'  , index_col = 0)

    # In[]
    hw4 = hw4_eigenvector(df, '5y'  ,'df')
    hw4._clean()
     
    
    returns = hw4.log_return()
    cov = hw4.covariance_matrix()
    # In[]
    value , vector =  np.linalg.eig(cov)
    # In[]
    member = hw4.find_cover(0.9)
    # In[]
    G = np.ones([2,100])
    G[1,17:] = 0
    # In[]
    u ,d, vt = hw4.svd()
    gcg = hw4.GCG(G)
    # In[]
    constant = np.array([1,0.1])
    a , d = hw4.portfolio(G,10,constant)
    # In[]
    '''plot the first PC '''
    df = hw4.df
    p = df.sum(1)/100
    w = list(member.values())[0]  
    aaaa = df @ (w / sum(w) )
    aaaa.plot(label = 'first')
    p.plot(label = 'sp500')
    plt.legend()
