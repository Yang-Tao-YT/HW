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
        '''find the loading'''
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
    
    def recover_from_PC (self, vector):
        '''recover the daily reutrn from eigenvector'''
        returns = self.log_return()
        PC = returns @ vector #calculate the principle component 
        re_hat = PC @ vector.T
        return re_hat
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
        c_inverse = u @ np.linalg.inv(d) @ vt
        
        gcg = G @ c_inverse @ G.T
        return gcg,c_inverse
    
    def portfolio(self, G,a,constant):
        '''find the weight of best portfolio'''
        cons = np.array([1,0.1])
        gcg, c_inverse = self.GCG(G)
        gcg_inverse = np.linalg.inv(gcg)
        mean = self.log_return().mean()
        lamda =gcg_inverse @ (G @ c_inverse @ mean - 2 * a * cons)
        weight  = 1 /(2 * a) * c_inverse @ (mean - G.T @ lamda)
        return weight
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
    '''find the eigenvector that cover for 90%'''
    member = hw4.find_cover(0.9) 
    # In[]
    G = np.ones([2,100])
    G[1,17:] = 0
    # In[]
    u ,d, vt = hw4.svd()
    gcg = hw4.GCG(G)
    # In[]
    constant = np.array([1,0.1])
    weight = hw4.portfolio(G,20,constant)
    # In[]
    '''plot the prediction '''

    vector = list(member.values())
    vector = np.matrix(vector).T
    returns_hat = hw4.recover_from_PC(vector)
    port_r = returns.sum(1)/100
    port_r_hat = returns_hat.sum(1)/100
    port_r.plot(fontsize = 20 , figsize = [20,15] , label = 'portfolio' )
    port_r_hat.plot(fontsize = 20 , figsize = [20,15] , label = 'portfolio_hat' , )
    plt.legend(fontsize = 20)
    plt.xlabel('date' , fontsize = 20)
    plt.title('daily return vs prediction from the 90% eigenvector' , fontsize = 20)
    plt.show()
    # In[]
    '''plot the difference'''
    diff = (port_r - port_r_hat)**2
    diff.plot(figsize = [20,15] , fontsize = 20 , label = 'difference between predict and return')
    plt.legend(fontsize = 20)
    plt.xlabel('Date' , fontsize = 20)
    plt.title('Daily return vs prediction from the 90% eigenvector' , fontsize = 20)
    plt.show()
    # In[]
    '''plot the best portfolio'''
    price = hw4.df.sum(1)/100
    price_port = hw4.df @ weight
    price.plot(label = 'marekt' ,  figsize = [20,15] , fontsize = 20)
    price_port.plot(label = 'portfolio' , figsize = [20,15] ,  fontsize = 20)
    plt.xlabel('Date', fontsize = 20)
    plt.title('market vs portfolio' , fontsize = 20)
    plt.legend()