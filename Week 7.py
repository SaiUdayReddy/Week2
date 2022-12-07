#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run the imports we need

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mnist = np.genfromtxt('MNIST_Shortened (1).csv', delimiter=',',
                     skip_header=1)

# Define X and y

X = mnist[:,0:784]
y = mnist[:,-1]

# check dimensions of X

X.shape


# In[3]:



plt.imshow(X[0].reshape(28,28),cmap='gray_r')


# In[4]:


plt.figure(figsize=(8, 12))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(X[np.random.randint(0,6000)].reshape(28,28),cmap='gray_r')
    plt.axis('off')

plt.show()


# In[5]:


# split data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=0, stratify=y)


# In[8]:


from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0)
X_train_fs = variance_selector.fit_transform(X_train)
X_test_fs = variance_selector.transform(X_test)
print(f"{X_train.shape[1]-X_train_fs.shape[1]} features have been removed, {X_train_fs.shape[1]} features remain")


# In[9]:


# we can use the get_support function to see which features have been droped

selected_features = variance_selector.get_support()

selected_features = selected_features.reshape(28,28)

# Visualise which pixels have been dropped

sns.heatmap(selected_features,cmap='rocket')


# In[12]:


# Use the SelectKBest selector from sklearn to select the k features with the best scores on a selected test statistic
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=200)
X_train_fs = selector.fit_transform(X_train_fs, y_train)
X_test_fs = selector.transform(X_test_fs)


# In[15]:


# Create boolean array for all features
new_features_indices = variance_selector.get_support(indices=True)[selector.get_support()]
new_features_boolean = np.isin(np.arange(784), new_features_indices)


# In[16]:


# Reshape and plot as a heatmap
sns.heatmap(new_features_boolean.reshape(28,28),cmap='rocket')


# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix
from sklearn.feature_selection import RFECV


# In[19]:


# Standardise data before passing to model
scaler = StandardScaler()
X_train_fs = scaler.fit_transform(X_train_fs)
X_test_fs = scaler.transform(X_test_fs)


# In[21]:


rf = RandomForestClassifier(random_state=0) # Use RandomForestClassifier as the best model
rfecv = RFECV(rf, cv=3, step=5)
X_train_fs = rfecv.fit_transform(X_train_fs, y_train)
X_test_fs = rfecv.transform(X_test_fs)
print(f"Number of remaining features: {X_train_fs.shape[1]}")


# In[28]:


plt.figure(figsize=(15, 6))
plt.title('Number of Features Included vs Accuracy')
plt.xlabel('Number of Features Included')
plt.ylabel('Model Accuracy')
plt.plot(np.linspace(0,200,41), rfecv.cv_results_['mean_test_score'])
plt.show()


# In[29]:


rf_selectedfeatures = RandomForestClassifier()
rf_selectedfeatures.fit(X_train_fs, y_train)


# In[30]:


# Make predictions on the test data
y_pred = rf_selectedfeatures.predict(X_test_fs)
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)*100:.2f}%")
cm = confusion_matrix(y_test,y_pred)
ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
plt.show()


# In[34]:


get_ipython().system('pip install imblearn')


# In[46]:


import numpy as np
import pandas as pd
import imblearn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:




