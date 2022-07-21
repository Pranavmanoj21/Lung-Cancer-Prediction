#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv('lung_cancer_data.csv')
data


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


data.duplicated().sum()    # sum of duplicate values


# In[10]:


data = data.drop_duplicates()

data.shape


# In[11]:


data.isnull().sum()


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


from sklearn import preprocessing


# In[15]:


label_encoder = preprocessing.LabelEncoder()  #LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels to numerical labels.
label_encoder


# In[16]:


data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])


# In[17]:


data.info()


# In[18]:


data.nunique()  # The nunique() method returns the number of unique values for each column.


# In[19]:


data.columns


# next step is to remove columns with continuous data and keep only columns with categorised

# In[20]:


data_new = data[['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']]


# In[21]:


data_new.columns


# In[22]:


data_new.nunique()


# In[23]:


for i in data_new.columns:
    plt.figure(figsize = (15, 6))
    sns.countplot(data_new[i], data = data_new,
                 palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()


# In[24]:


for i in data_new.columns:
    data_new.value_counts().plot(kind = 'pie',
                                figsize = (8, 8),
                                autopct = '%1.1f%%')
    plt.xticks(rotation = 90)
    plt.show()


# In[25]:


data_new['LUNG_CANCER'].unique()


# In[26]:


data_new['LUNG_CANCER'].value_counts()


# In[27]:


data_new['LUNG_CANCER'].value_counts() / len(data_new['LUNG_CANCER']) * 100


# In[28]:


plt.figure(figsize = (15, 6))
sns.histplot(data['AGE'])
plt.xticks(rotation = 90)
plt.show()


# In[29]:


data_new['GENDER'].unique()


# In[30]:


data_new['GENDER'].value_counts()


# In[31]:


data_new['GENDER'].value_counts() / len(data['GENDER']) * 100


# In[32]:


plt.figure(figsize = (15, 6))
sns.countplot('GENDER', data = data_new, hue = 'LUNG_CANCER',
             palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[33]:


plt.figure(figsize = (15, 6))
sns.countplot(x ='COUGHING', data = data_new, hue = 'LUNG_CANCER',
             palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[34]:


plt.figure(figsize = (15, 6))
sns.countplot(x = 'YELLOW_FINGERS', data = data_new, hue = 'LUNG_CANCER',
             palette = 'hls')
plt.legend(['Has cancer', 'Does not have cancer'])
plt.xticks(rotation = 90)
plt.show()


# In[35]:


plt.figure(figsize = (15, 6))
sns.countplot(x = 'ANXIETY', data = data_new, hue = 'SHORTNESS OF BREATH')
plt.legend('Has cancer', 'Does not have cancer')
plt.xticks(rotation = 90)
plt.show()


# In[36]:


plt.figure(figsize = (15, 6))
sns.distplot(data['AGE'])
plt.xticks(rotation = 90)
plt.show()


# In[37]:


corrmat = data_new.corr()
corrmat


# In[38]:


cmap = sns.diverging_palette(260, -10, s = 50, l = 75, n = 6,
                            as_cmap = True)
plt.subplots(figsize = (18, 18))
sns.heatmap(corrmat, cmap = cmap, annot = True, square = True)
plt.show()


# In[39]:


x = data_new.drop('LUNG_CANCER', axis = 1)    # because we need to predict this in the end. this should not affect the final prediction
y = data_new['LUNG_CANCER']


# In[40]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.25, 
                                                    random_state = 0)   # The random state hyperparameter in the train_test_split() function controls the shuffling process. With random_state=0 , we get the same train and test sets across different executions.


# In[41]:


from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[42]:


y_pred =  classifier.predict(x_test)


# In[43]:


from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, median_absolute_error, accuracy_score, f1_score


# In[44]:


from sklearn.metrics import plot_roc_curve


# In[45]:


print("Mean absolute error is: ", (mean_absolute_error(y_test, y_pred)))
print("Mean squared error is: ", mean_squared_error(y_test, y_pred))
print("Median absolute error is: ", median_absolute_error(y_test, y_pred))
print("Accuracy is: ", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("f1 score: ", round(f1_score(y_test, y_pred, average = "weighted") * 100, 2), "%")


# In[47]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
print('Confusion matrix : \n',matrix)
tp, fn, fp, tn = confusion_matrix(y_test,y_pred,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)


# In[ ]:




