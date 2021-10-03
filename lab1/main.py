#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import bernoulli as brnl
import os


# In[140]:


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
    
    
    def calc_Bayes(self, PC, PM_C):
        temp = np.zeros(PC.shape[0], dtype=np.int64)
        temp2 = np.zeros((PC.shape[0], PC.shape[0]), dtype=np.int64)
        loss = 0
#         print(PC.shape)
#         PC_T = PC.T
#         print(PC.shape)
        PM_C_T = PM_C.T
        for i in range(PM_C_T.shape[0]):
            index = np.where(PM_C_T[i] == PM_C_T[i].max())[0]
            temp[i] = index[0] if len(index) == 1 else index[-1]
            temp2[i][temp[i]] = 1
#             print(type(temp[i]))
            print(f'M[{temp[i]}] ===> C[{i}]')
#         print(temp)
#         print(temp2)
        
        
        
        for i in range(PC.shape[0]):
#             print(temp[i])
            loss += PC[i] * (1 - PM_C[int(temp[i])][i])
        print(f'Loss: {loss}')
        return temp2

    
    def calc_stochastic(self, PC, PM_C):
        loss = 0
        temp = np.zeros((PC.shape[0], PC.shape[0]), dtype=np.float64)
        PM_C_T = PM_C.T
        for i in range(PM_C.shape[0]):
            indexes = np.where(PM_C_T[i] == PM_C_T[i].max())[0]
            for index in indexes:
                temp[i][int(index)] = 1 / len(indexes)
#                 print(index)
        
#         print(temp)
        
        for i in range(20):
            l1 = 0
            for j in range(20):
                l1 += PM_C_T[i][j] * temp[i][j]
            loss += (1 - l1) * PC[i]
        print(f'Loss: {loss}')
        return temp
        
        
        
        
        
        
#         loss = 0
#         PM_C_T = PM_C.T

#         temp1 = np.zeros((PC.shape[0], PC.shape[0]))
#         temp2 = np.zeros(PC.shape[0])
#         temp3 = np.zeros_like(temp1)

#         for i in range(PC.shape[0]):
#             indexes = np.where(PM_C_T[i] == PM_C_T[i].max())[0]
# #             print(indexes, len(indexes))
#             for index in indexes:
#                 temp1[i][index] = 1 / len(indexes)
#             temp2[i] = len(indexes)
#             print("Количество максимумов в столбце [",i,"] =", len(indexes))
# #         print(temp1)


# #         for i in range(PC.shape[0]):
# #             t = temp2[i]
# #             index = np.where(temp1[i] != 0)
# #             print(index)
# #             for index in indexes:
# #                 print(index)
# #             temp3[i][index] = brnl.rvs(1 / t)
# #             print(i)
        
# #             print(np.where(temp3[i] != 0))
# #         print(temp3)
#         for i in range(PC.shape[0]):
#             t = temp2[i]
#             for j in range(PC.shape[0]):
#                 if temp1[i][j] != 0:
#                     temp3[i][j] = brnl.rvs(1 / t)
#                     t -= 1
#                 if temp3[i][j]:
#                     break
#         for i in range(PC.shape[0]):
#             print(f'M[{np.where(temp3[i] != 0)[0][0]}] ===> C[{i}]')
#         print(temp3)
#         for i in range(20):
#             l1 = 0
#             for j in range(20):
#                 l1 += PM_C_T[i][j] * temp1[i][j]
#             loss += (1 - l1) * PC[i]
#         print(f'Loss: {loss}')
        
        
    
    def __repr__(self):
        return 'Lab1 is not so clear :('


# # 10 Вариант

# In[141]:


a = Lab1('vars', 10)


# In[142]:


PC = a.calc_PC()
print('Распределение шифртекста:')
pd.DataFrame(PC)


# In[143]:


assert PC.sum().round(10) == 1.0, 'Should be 1.0'


# In[144]:


PMC = a.calc_PMC()
print('Распеделение открытых текстов и шифртекстов:')
pd.DataFrame(PMC)


# In[145]:


assert PMC.sum().round(10) == 1.0, 'Should be 1.0'


# In[146]:


PM_C = a.calc_PM_C(PC, PMC)
print('Условное распределение:')
pd.DataFrame(PM_C)


# In[147]:


assert PM_C.sum().round() == 20.0, 'Should be 20.0'


# In[150]:


print('Детерминистическая решающая функция и потери:')
pd.DataFrame(a.calc_Bayes(PC, PM_C))


# In[151]:


print('Стохастическая функция и потери:')
pd.DataFrame(a.calc_stochastic(PC, PM_C))


# # 6 Вариант

# In[152]:


b = Lab1('vars', '06')


# In[153]:


PC = b.calc_PC()
print('Распределение шифртекста:')
pd.DataFrame(PC)


# In[154]:


assert PC.sum().round(10) == 1.0, 'Should be 1.0'


# In[155]:


PMC = b.calc_PMC()
print('Распеделение открытых текстов и шифртекстов:')
pd.DataFrame(PMC)


# In[156]:


assert PMC.sum().round(10) == 1.0, 'Should be 1.0'


# In[157]:


PM_C = b.calc_PM_C(PC, PMC)
print('Условное распределение:')
pd.DataFrame(PM_C)


# In[158]:


assert PM_C.sum().round() == 20.0, 'Should be 20.0'


# In[159]:


print('Детерминистическая решающая функция и потери:')
pd.DataFrame(b.calc_Bayes(PC, PM_C))


# In[160]:


print('Стохастическая функция и потери:')
pd.DataFrame(b.calc_stochastic(PC, PM_C))


# In[ ]:




