#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
from scipy.stats import bernoulli as brnl
import os


# In[85]:


class Lab1:
    def __init__(self, var_path, var):
        prob_filename = f'prob_{str(var)}.csv'
        table_filename = f'table_{str(var)}.csv'
        self.prob_path = os.path.join(var_path, prob_filename)
        self.table_path = os.path.join(var_path, table_filename)
        self.prob = pd.read_csv(self.prob_path, header=None)
        self.table = pd.read_csv(self.table_path, header=None)
    
    
    def calc_PC(self):
        PC = np.zeros(self.table.shape[0], dtype=np.float64)
        for i in range(self.table.shape[0]):
            X, Y = np.where(self.table == i)
            for x, y in zip(X, Y):
                PC[i] += self.prob.iloc[0, y] * self.prob.iloc[1, x]
        return PC
    
    
    def calc_PMC(self):
        PMC = np.zeros((self.table.shape[0], self.table.shape[0]), dtype=np.float64)
        for i in range(self.table.shape[0]):
            X, Y = np.where(self.table == i)
            for x, y in zip(X, Y):
                PMC[y][i] += self.prob.iloc[0, y] * self.prob.iloc[1, x]
        return PMC
    
    
    def __repr__(self):
        return 'Lab1 is not so clear :('


# In[86]:


a = Lab1('vars', 10)


# In[87]:


res = a.calc_PC()


# In[96]:


assert res.sum() == 1.0, 'Should be 1.0'


# In[97]:


res1 = a.calc_PMC()


# In[98]:


#get_ipython().run_cell_magic('timeit', '', 'sum(sum(res1))')


# In[99]:


#get_ipython().run_cell_magic('timeit', '', 'res1.sum()')


# In[100]:


assert res1.sum() == 1.0, 'Should be 1.0'


# In[ ]:




