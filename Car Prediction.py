#!/usr/bin/env python
# coding: utf-8

# # Business Problem

# In[1]:


import numpy as np


# In[30]:


import pandas as pd
import seaborn as sns


# In[59]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Dataset

# In[26]:


df=pd.read_csv('car data.csv')


# In[27]:


df.head()


# In[28]:


df.info()


# In[32]:


df.shape


# In[54]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())


# In[34]:


df.isnull().sum()


# In[38]:


df.columns


# In[39]:


final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[40]:


final_dataset.head()


# In[41]:


final_dataset['Current_Year']=2020


# In[42]:


final_dataset.head()


# In[43]:


final_dataset['No_Year']=final_dataset['Current_Year']-final_dataset['Year']


# In[44]:


final_dataset.head()


# In[46]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[47]:





# In[50]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[51]:





# In[52]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[53]:


final_dataset.head()


# In[55]:


final_dataset.corr()


# In[56]:


sns.boxplot(final_dataset)


# In[57]:


sns.pairplot(final_dataset)


# In[58]:


sns.heatmap(final_dataset)


# In[66]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[67]:


x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[68]:


x.head()


# In[69]:


y.head()


# In[70]:


from sklearn.ensemble import ExtraTreesRegressor


# In[71]:


model=ExtraTreesRegressor()


# In[72]:


model.fit(x,y)


# In[74]:


print(model.feature_importances_)


# In[77]:


feat_importances=pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[78]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[79]:


x_train.shape


# In[99]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[100]:


n_estimators=[int(x)for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)


# In[91]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = 1,2,5,10


# In[92]:


from sklearn.model_selection import RandomizedSearchCV


# In[93]:


random_grid ={'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf}
print(random_grid)


# In[102]:


rf=RandomForestRegressor()


# In[105]:


rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[106]:


rf_random.fit(x_train,y_train)


# In[109]:


predictions=rf_random.predict(x_test)


# In[110]:


predictions


# In[111]:


sns.distplot(y_test-predictions)


# In[113]:


plt.scatter(y_test,predictions)


# In[114]:


import pickle


# In[115]:


file= open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random,file)


# In[ ]:




