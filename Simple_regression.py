#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[9]:


#scatter plot to gauge the results

df = pd.read_excel('data OLS.xlsx', index_col='Year')

df.plot.scatter(x = 'excess return on market', y = 'Excess return on fund')


# In[13]:


# create a linear regression 
X = df[["excess return on market"]]
y = df[["Excess return on fund"]]

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))


# In[15]:


#create another plot with regression line included

plt.scatter(X, y, color = 'red')
plt.plot(X, reg.predict(X), color = 'blue')
plt.title('mark1 vs mark2')
plt.xlabel('mark1')
plt.ylabel('mark2')
plt.show()

