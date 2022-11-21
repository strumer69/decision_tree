#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('kyphosis.csv')


# In[10]:


df.head()


# In[11]:


df.info()


# In[36]:


df.describe()


# # EDA
# We'll just check out a simple pairplot for this small dataset.

# In[42]:


sns.displot(x=df['Age'],hue=df['Kyphosis'])


# In[13]:


sns.displot(x=df['Age'])


# In[37]:


present_values=df[df['Kyphosis']=='present']


# In[39]:


sns.displot(present_values['Age'],color='red')


# In[16]:


absent=df[df['Kyphosis']=='absent']


# In[17]:


sns.displot(absent['Age'],bins=7,color='red')


# In[49]:


g = sns.FacetGrid(data=df,col='Kyphosis')
g.map(plt.hist,'Number')


# In[46]:


df.describe()


# # Train Test Split
# Let's split up the data into a training set and a test set!

# In[19]:


from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# # Decision Trees
# We'll start just by training a single decision tree.

# In[20]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# # Prediction and Evaluation
# Let's evaluate our decision tree.

# In[21]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[22]:


print(confusion_matrix(y_test,predictions))


# In[23]:


len(y_test)


# In[24]:


sum(y_test=='absent')


# In[25]:


sum(y_test=='present')


# # Random Forests
# Now let's compare the decision tree model to a random forest.

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[27]:


rfc_pred = rfc.predict(X_test)


# In[28]:


print(confusion_matrix(y_test,rfc_pred))


# In[29]:


print(classification_report(y_test,rfc_pred))


# # Logistic Regression model
# Now let's compare the decision tree model and random forest to Logistic Regression model.

# In[31]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[33]:


predictions = logmodel.predict(X_test)
confusion_matrix(y_test,predictions)


# In[34]:


print(classification_report(y_test,predictions))


# In[ ]:




