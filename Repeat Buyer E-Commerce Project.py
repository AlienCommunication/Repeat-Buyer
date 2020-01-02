#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc


# In[36]:


train = pd.read_csv("train_format1.csv")
user_log = pd.read_csv("user_log_format1.csv")
user_info = pd.read_csv("user_info_format1.csv")
test = pd.read_csv("test_format1.csv")


# # Analysing Format1 Files

# In[37]:


train.head()


# In[38]:


train.shape


# ####  User Log

# In[39]:


user_log.head()


# In[40]:


user_log.shape


# In[41]:


user_log.isnull().sum()


# #### User Info

# In[42]:


user_info.head()


# In[43]:


user_info.shape


# In[44]:


user_info.isnull().sum()


# # Format Files 2

# In[45]:


Test2 = pd.read_csv("test_format2.csv")


# In[46]:


Train2 = pd.read_csv("train_format2.csv")


# In[47]:


Test2.head()


# In[48]:


Train2.head()


# In[49]:


train['origin'] = 'train'
test['origin'] = 'test'


# In[50]:


matrix = pd.concat([train, test], ignore_index=True, sort=False)
matrix.drop(['prob'], axis=1, inplace=True)
matrix = matrix.merge(user_info, on='user_id', how='left')


# In[51]:


matrix.head()


# In[52]:


user_log.rename(columns={'seller_id':'merchant_id'}, inplace=True)


# In[53]:


user_log['user_id'] = user_log['user_id'].astype('int32')
user_log['merchant_id'] = user_log['merchant_id'].astype('int32')
user_log['item_id'] = user_log['item_id'].astype('int32')
user_log['cat_id'] = user_log['cat_id'].astype('int32')
user_log['brand_id'].fillna(0, inplace=True)
user_log['brand_id'] = user_log['brand_id'].astype('int32')
user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%H%M')


# In[59]:


matrix['age_range'].fillna(0, inplace=True)
matrix['gender'].fillna(2, inplace=True)
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'] = matrix['gender'].astype('int8')
matrix['label'] = matrix['label'].astype('str')
matrix['user_id'] = matrix['user_id'].astype('int32')
matrix['merchant_id'] = matrix['merchant_id'].astype('int32')


# In[62]:


groups = user_log.groupby(['user_id'])
temp = groups.size().reset_index().rename(columns={0:'u1'})
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds/3600
matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u7', 1:'u8', 2:'u9', 3:'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')


# In[64]:


groups = user_log.groupby(['merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'m1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'user_id':'m2',
    'item_id':'m3', 
    'cat_id':'m4', 
    'brand_id':'m5'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m6', 1:'m7', 2:'m8', 3:'m9'})
matrix = matrix.merge(temp, on='merchant_id', how='left')


# In[66]:


temp = Train2[Train2['label']==-1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m10'})
matrix = matrix.merge(temp, on='merchant_id', how='left')


# In[67]:


groups = user_log.groupby(['user_id', 'merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'um1'})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'item_id':'um2',
    'cat_id':'um3',
    'brand_id':'um4'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
    0:'um5',
    1:'um6',
    2:'um7',
    3:'um8'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['time_stamp'].agg([('frist', 'min'), ('last', 'max')]).reset_index()


# In[68]:


temp['um9'] = (temp['last'] - temp['frist']).dt.seconds/3600
temp.drop(['frist', 'last'], axis=1, inplace=True)
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')


# In[69]:


matrix['r1'] = matrix['u9']/matrix['u7'] 
matrix['r2'] = matrix['m8']/matrix['m6'] 
matrix['r3'] = matrix['um7']/matrix['um5']


# In[70]:


matrix.fillna(0, inplace=True)


# In[71]:


temp = pd.get_dummies(matrix['age_range'], prefix='age')
matrix = pd.concat([matrix, temp], axis=1)
temp = pd.get_dummies(matrix['gender'], prefix='g')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop(['age_range', 'gender'], axis=1, inplace=True)


# In[72]:


#train、test-setdata
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

del temp, matrix
gc.collect()


# In[73]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import xgboost as xgb


# In[74]:


X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.3)


# # Random Forest Model

# In[75]:


rf_clf = RandomForestClassifier(
    oob_score=True, 
    n_jobs=-1, 
    n_estimators=1000, 
    max_depth=10, 
    max_features='sqrt')

rf_clf.fit(X_train, y_train)


# #  XGBOOST MODEL

# In[76]:




model = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42
    
)

model.fit(
    X_train, 
    y_train,
    eval_metric='auc',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10
)


# In[77]:


prob = model.predict_proba(test_datat)
submission['prob'] = pd.Series(prob[:,1])
submission.dorp(['origin'], axis=1, inplace=True)
submission.to_csv('RepeatBuyer.csv', index=False)


# In[ ]:




