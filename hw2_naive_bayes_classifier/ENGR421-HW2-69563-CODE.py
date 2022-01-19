#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necc packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#safelog for not encountring issues with 0
def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


#read given data
data_set = np.genfromtxt("hw02_images.csv", delimiter = ",")
labels = np.genfromtxt("hw02_labels.csv", delimiter = ",").astype(int)


# In[3]:


#split our data set into training and test sets
train_indices = np.arange(0,30000)
test_indices = np.arange(30000,35000)
X_train = data_set[train_indices,:]
y_train = labels[train_indices]
X_test = data_set[test_indices,:]
y_test = labels[test_indices]


# In[4]:


#necc variables
N = data_set.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]
K = np.max(labels).astype(int)
D = data_set.shape[1]


# In[5]:


def mu_c(X, y, c):
    #Returns a matrix with shape (D,)
    #number of data points from class c:
    N_c = np.sum(y == c)
    #sum of x vals
    summ= np.array(np.sum([X[i] *(1 if y[i] == c else 0) for i in range (X.shape[0])] , axis=0))
    return summ/N_c
#A (DxK) matrix
sample_means = np.stack([mu_c(X_train, y_train, c+1) for c in range(K)]).T
sample_means


# In[14]:


def sigma_c(X, y, c):
    #Returns deviation of class c as a matrix with shape(D,)
    mean = mu_c(X,y,c)
    summ = np.sum([(1 if y[i]==c else 0)* (X[i]-mean)**2 for i in range (X.shape[0])],axis=0)
    avg = summ/ np.sum(y==c)
    return np.sqrt(avg)
#A (DxK)matrix
sample_deviations = np.stack([sigma_c(X_train, y_train, c+1) for c in range(K)]).T
sample_deviations


# In[7]:


def prior(X,y,c):
    return np.sum(y == c) / X.shape[0]

class_priors= np.stack([prior(X_train, y_train, c+1) for c in range(K)])
class_priors


# In[15]:


#score function for Naive Bayes classfier for cont's features
def score_c(x, c):
    return -(np.sum(safelog(sample_deviations[:, c])) + np.sum(((x - sample_means[:, c]) ** 2) / (2 * (sample_deviations[:, c] ** 2)))) + safelog(class_priors[c]) 


# In[16]:


#predict for each data point on training set the class with highest score 
def predict_class(x):
    return np.argmax(np.array([score_c(x, c) for c in range(K)])) + 1


# In[17]:


#predict for each data point in training set
y_predict = np.array([predict_class(X_train[i]) for i in range(N_train)])


# In[18]:


#create confusion matrix for y_train and predicted data on training set
confusion_matrix = pd.crosstab(y_predict, y_train, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[19]:


#predict for each data point on test set the class with highest score 
y_test_predict =  np.array([predict_class(X_test[i]) for i in range(N_test)])


# In[20]:


#create confusion matrix for y_test and predicted data on test set
confusion_matrix = pd.crosstab(y_test_predict, y_test, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[ ]:




