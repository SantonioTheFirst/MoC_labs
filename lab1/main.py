#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import bernoulli as brnl
import os


# In[2]:


class Lab1:
    def __init__(self, var_path, var):
        prob_filename = f'prob_{str(var)}.csv'
        table_filename = f'table_{str(var)}.csv'
        self.prob_path = os.path.join(var_path, prob_filename)
        self.table_path = os.path.join(var_path, table_filename)
        self.prob = pd.read_csv(self.prob_path, header=None)
        self.table = pd.read_csv(self.table_path, header=None)
    
    
    def __repr__(self):
        return 'Lab1 is not so clear :('


# In[3]:


a = Lab1('vars', 10)


# In[ ]:




