#!/usr/bin/env python
# coding: utf-8

# # # Real Estate Price Predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv('data.csv',encoding='utf-16', sep='\t')


# In[3]:


housing.head(
)


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe(
)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50, figsize=(20,15))


# ## Train Test Splitting

# In[10]:


#learning Purpose
import numpy as np
def split_train(data, test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


train_set, test_set =split_train(housing, 0.2)


# In[12]:


#print(f"Rows in train set: {len(train_set)}\nRows in Test set: {len(train_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in Test set: {len(train_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_train_set.info()


# In[16]:


housing=strat_train_set.copy()


# # Looking For Coorelation

# In[17]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[ ]:





# In[18]:


from pandas.plotting import scatter_matrix
attributes= ['RM','ZN','MEDV','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))


# In[19]:


housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)


# # Trying out attributes combinations

# In[20]:


housing['TAXRM']= housing['TAX']/housing['RM']
housing['TAXRM']


# In[21]:


housing.head()


# In[22]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


housing.plot(kind='scatter',x='TAXRM',y='MEDV',alpha=0.8)


# In[24]:


housing= strat_train_set.drop('MEDV', axis=1)
housing_labels= strat_train_set['MEDV'].copy()


# ## Missing Attributes

# In[25]:


#To take care of miisng attributes, you have three options:
 #   1. Get rid of misssing attribute
  #  2. Get rid of the whole attribute
   # 3. Set the value to same value(0, mean or median)


# In[26]:


housing.head()


# In[27]:


median=housing['RM'].median()


# In[28]:


median


# In[29]:


housing['RM'].fillna(median)


# In[30]:


housing.shape


# In[31]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy= 'median')
imputer.fit(housing)


# In[32]:


imputer.statistics_


# In[33]:


X= imputer.transform(housing)


# In[34]:


housing_tr=pd.DataFrame(X, columns=housing.columns)


# In[35]:


housing_tr.describe()


# ## Scikit-Learn Design

# Primarily, there are three types of objects:
# 1. Estimators- It estimates some parameters based on a datatset.Eg: Imputer
# It has a fit method and a transform method.
# Fit Method- Fits the dataset and calculates internal parameters
#     
# 2. Transformers- It takes input and returns output based on the learnings from fit(). 
# It also has a convenience functoion called fit_transform which fits and then transforms.
# 
# 3. Predictors- LinearRegression model is an example of predictor. fit() and predict() are two common functions. 
# It also gives score() function which will evaluate the predictions.

# ##Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-Max Scaling(Normalization)
# (value-min)/(max-min)
# Sklearn provides a class called MinmaxScaler for this
# 
# 2.Standardization
# value-min/std
# Sklearn provides a class called StandardScaler for this

# ## Creating Pipeline

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])


# In[37]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[38]:


housing_num_tr.shape


# 
# ## Selecting A design model for Dragon Real Estate

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model =RandomForestRegressor()
#odel =LinearRegression()
model.fit(housing_num_tr, housing_labels)


# In[40]:


some_data= housing.iloc[:5]


# In[41]:


some_labels= housing_labels.iloc[:5]


# In[42]:


prepared_data= my_pipeline.transform(some_data)


# In[43]:


model.predict(prepared_data)


# In[44]:


list(some_labels)


# ## Evaluating the model

# In[45]:


from sklearn.metrics import mean_squared_error
housing_predictions= model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)


# In[46]:


rmse


# # Using Better Validation Technique- CRoss validation

# In[47]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores= cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores= np.sqrt(-scores)


# In[48]:


rmse_scores


# In[49]:


def print_scores(scores):
    print("Scores are: ", scores)
    print("Mean: ",scores.mean())
    print("Standard deviations: ",scores.std())


# In[50]:


print_scores(rmse_scores)


# ## Saving The Model

# In[52]:


from joblib import dump, load
dump(model, 'Real Estate.joblib')


# ## Testing the Model on test data

# In[58]:


x_test=strat_test_set.drop('MEDV', axis=1)
y_test=strat_test_set['MEDV'].copy()
x_test_prepared= my_pipeline.transform(x_test)
final_predictions= model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(y_test))


# In[56]:


final_rmse

