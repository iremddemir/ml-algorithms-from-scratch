#!/usr/bin/env python
# coding: utf-8

# In[28]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# In[29]:


#read given data
data_set = np.genfromtxt("hw06_images.csv", delimiter = ",")
labels = np.genfromtxt("hw06_labels.csv", delimiter = ",").astype(int)


# In[30]:


#split our data set into training and test sets
train_indices = np.arange(0,1000)
test_indices = np.arange(1000,5000)
X_train = data_set[train_indices,:]
y_train = labels[train_indices]
X_test = data_set[test_indices,:]
y_test = labels[test_indices]
class_size = np.amax(y_train)


# In[72]:


#X: data set labels to be predicted
#X_train : training set y: labels of X_train
#C,s : parameters
def one_vs_all(X, X_train, y, C, s):
    N = X.shape[0]
    y_pred = np.zeros((class_size, N))
    for i in range (class_size):
        class_label = i+1
        bi_y = binary_y(y, class_label)
        K, alpha, w0 = svm_train(X,X_train, bi_y, C, s)
        f_predicted= f_predict(bi_y,K,alpha,w0)
        y_pred[i] = np.reshape(f_predicted, N)
    y_predicted = np.argmax(y_pred, axis =0)+1

    return y_predicted
    
#convert labels with given label to 1 not given label to -1 for preperaring one vs all
def binary_y(y, class_label):
    N = y.shape[0] 
    bi_y = np.zeros(N)
    for i in range (N):
        if y[i] == class_label:
            bi_y[i]=1
        else:
            bi_y[i]=-1
    return bi_y
#calculate f_predicted
def f_predict(y,K,alpha,w0):
    f_predicted = np.matmul(K, y[:,None] * alpha[:,None]) + w0
    return f_predicted
#train svm using X and images,labels, and learning parameters
def svm_train(X, X_train, y, C, s, epsilon = 1e-3):
    K_train = gaussian_kernel(X_train,X_train,s)
    yyK = hadamard(y, K_train)
    w0 , alpha = dual_problem(C,epsilon, yyK,y)
    K = gaussian_kernel(X, X_train, s)
    return K, alpha, w0
# define Gaussian kernel function as similarity metric 
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)
# define hadamard mult as yy^TK
def hadamard(y,K): 
    yyK = np.matmul(y[:,None], y[None,:]) * K
    return yyK
#solve dual problem matrix-vector form
def dual_problem(C,epsilon, yyK, y):
    N = y.shape[0]
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = cvx.matrix(np.vstack((np.zeros((N, 1)), C * np.ones((N, 1)))))
    A = cvx.matrix(1.0 * y[None,:])
    b = cvx.matrix(0.0)
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    return w0, alpha


# In[73]:


#Train algorithm using training set & C=10 s = 10
C = 10
s = 10
#Training Performance:
N_train = y_train.shape[0]
y_predicted_train = one_vs_all(X_train,X_train, y_train,C, s)
confusion_matrix = pd.crosstab(np.reshape(y_predicted_train, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)




# In[75]:


#Test Performance:
N_test = y_test.shape[0]
y_predicted_test = one_vs_all(X_test,X_train, y_train,C, s)
confusion_matrix = pd.crosstab(np.reshape(y_predicted_test, N_test), y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(confusion_matrix)


# In[83]:


#define perc accuracy as the number of accurately predicted over full size
def perc_accuracy(y_hat, y):
    accurate = 0
    N = y.shape[0]
    for i in range(N):
        if y_hat[i] == y[i]:
            accurate += 1
    return accurate / N
C = np.array([0.1, 1,10,100,1000])
train_accuracy = []
test_accuracy = []
#training performance for different Cs
for c in C:
    y_predicted_train=one_vs_all(X_train,X_train, y_train,c, s = 10)
    accuracy = perc_accuracy(y_predicted_train, y_train)
    train_accuracy.append(accuracy)
#test performance for different Cs
for c in C:
    y_predicted_test=one_vs_all(X_test,X_train, y_train,c, s = 10)
    accuracy = perc_accuracy(y_predicted_test, y_test)
    test_accuracy.append(accuracy)


# In[95]:


#PLOT
#imported so that regularization parameters can have equally partioned in x-axis
from matplotlib import ticker

f = plt.figure(figsize = (10, 6))
f_x = f.add_subplot(1,1,1)
plt.xlabel("Regularization Parameter")
plt.ylabel("Accuracy")
plt.scatter(C,train_accuracy)
plt.plot(C,train_accuracy,label = "Training Set")
plt.scatter(C,test_accuracy)
plt.plot(C,test_accuracy,label = "Test Set")
f_x.set_xscale('symlog')
f_x.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
plt.legend(loc = "upper right")

plt.show()   


# In[81]:


train_accuracy


# In[85]:


test_accuracy


# In[ ]:




