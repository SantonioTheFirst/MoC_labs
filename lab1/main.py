#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import bernoulli as brnl
import os


# In[402]:


class Lab1:
    def __init__(self, var_path, var):
        prob_filename = f'prob_{str(var)}.csv'
        table_filename = f'table_{str(var)}.csv'
        self.prob_path = os.path.join(var_path, prob_filename)
        self.table_path = os.path.join(var_path, table_filename)
        self.prob = pd.read_csv(self.prob_path, header=None)
        self.table = pd.read_csv(self.table_path, header=None)
    
    
    def calc_PC(self):
        PC = np.zeros(self.table.shape[0])
        for i in range(self.table.shape[0]):
            X, Y = np.where(self.table == i)
            for x, y in zip(X, Y):
                PC[i] += self.prob.iloc[0, y] * self.prob.iloc[1, x]
        return np.around(PC, 4)
    
    
    def calc_PMC(self):
        PMC = np.zeros((self.table.shape[0], self.table.shape[0]))
        for i in range(self.table.shape[0]):
            X, Y = np.where(self.table == i)
            for x, y in zip(X, Y):
                PMC[y][i] += self.prob.iloc[0, y] * self.prob.iloc[1, x]
        return np.around(PMC, 4)
    
    
    def calc_PM_C(self, PC, PMC):
        PC = PC.T
        PM_C = np.zeros((PMC.shape[0], PMC.shape[0]))
        for i in range(PMC.shape[0]):
            PM_C[i] = PMC[i] / PC
        return np.around(PM_C, 4)
    
    
    def cal_Bayes(self, PC, PM_C):
        temp = np.zeros(PC.shape[0], dtype=np.int64)
        loss = 0
#         print(PC.shape)
#         PC_T = PC.T
#         print(PC.shape)
        PM_C_T = PM_C.T
        for i in range(PM_C_T.shape[0]):
            temp[i] = int(np.where(PM_C_T[i] == PM_C_T[i].max())[0][0])
#             print(type(temp[i]))
            print(f'M[{temp[i]}] ===> C[{i}]')
        
        for i in range(PC.shape[0]):
#             print(temp[i])
            loss += PC[i] * (1 - PM_C[int(temp[i])][i])
        print(f'Loss: {loss}')
        return temp
        
    
    def __repr__(self):
        return 'Lab1 is not so clear :('


# In[403]:


a = Lab1('vars', '01')


# In[404]:


PC = a.calc_PC()
PC


# In[405]:


assert res.sum().round(10) == 1.0, 'Should be 1.0'


# In[406]:


PMC = a.calc_PMC()


# In[407]:


assert PMC.sum().round(10) == 1.0, 'Should be 1.0'


# In[408]:


PM_C = a.calc_PM_C(PC, PMC)
PM_C


# In[409]:


assert PM_C.sum().round(3) == 20.0, 'Should be 20.0'


# In[410]:


a.cal_Bayes(PC, PM_C)

