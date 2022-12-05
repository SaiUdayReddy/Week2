#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[5]:


import pandas as pd


# In[16]:


df = pd.read_csv('Downloads\Car_Insurance.csv')


# In[18]:


df


# In[19]:


df.iloc[:,:]


# In[20]:


df.iloc[:,0]


# In[21]:


df.iloc[2,0]


# In[22]:


df.iloc[1,0]


# In[23]:


df.iloc[5,:]


# In[24]:


df['Age']


# In[25]:


df.Age


# In[29]:


df.iloc[0,1]


# In[30]:


df.iloc[:,0].unique()


# In[36]:


df.iloc[:,0].value_counts()


# In[38]:


df.iloc[:,0].isin(['Nissan'])


# In[42]:


df.iloc[:,0].isin(['Toyota'])


# In[44]:


df.iloc[:,0].isin(['Skoda'])


# In[45]:


df.iloc[:,1].sort_values()


# In[46]:


df.iloc[:,1].sort_values(ascending=False)


# In[47]:


df.iloc[:,1].isnull()


# In[48]:


df.iloc[:,1].isnull().values.any()


# In[49]:


df['Age'].min()


# In[50]:


df['Age'].max()


# In[51]:


df['Age'].mean()


# In[55]:


get_ipython().system('pip install scikit-learn --quiet')


# In[56]:


import sklearn


# In[57]:


df.iloc[:,:]


# In[58]:


df.isnull().sum()


# In[59]:


pwd


# In[60]:


from sklearn.impute import SimpleImputer


# In[61]:


# numerical variables
# strategies: mean, median, most_frequent, constant
import numpy as np
numImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numImputer = numImputer.fit(df[['Age','Mileage']])
new_serr=numImputer.transform(df[['Age', 'Mileage']])
new_serr


# In[62]:


# numerical variables
# strategies: mean, median, most_frequent, constant
import numpy as np
numImputer = SimpleImputer(missing_values=np.nan, strategy='median')
numImputer = numImputer.fit(df[['Age','Mileage']])
new_serr=numImputer.transform(df[['Age', 'Mileage']])
new_serr


# In[63]:


# numerical variables
# strategies: mean, median, most_frequent, constant
import numpy as np
numImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
numImputer = numImputer.fit(df[['Age','Mileage']])
new_serr=numImputer.transform(df[['Age', 'Mileage']])
new_serr


# In[64]:


# numerical variables
# strategies: mean, median, most_frequent, constant
import numpy as np
numImputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-1)
numImputer = numImputer.fit(df[['Age','Mileage']])
new_serr=numImputer.transform(df[['Age', 'Mileage']])
new_serr


# In[65]:


# numerical variables
# strategies: mean, median, most_frequent, constant
import numpy as np
numImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numImputer = numImputer.fit(df[['Age','Mileage']])
new_serr=numImputer.transform(df[['Age', 'Mileage']])
new_serr


# In[ ]:




