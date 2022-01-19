#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necc. packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


np.random.seed(421)
#set mean parameters as given in instructions
class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                        [+2.5, -2.0]])
#set covariance parameters as given in instructions
class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]],
                              [[+1.2, -0.8], 
                               [-0.8, +1.2]]])
# set sample sizes as given in instructions
class_sizes = np.array([120,80, 100])


# In[3]:


# generate data
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2, points3))

# generate corr. labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))


y_truth = y.astype(int)

# get important 
K = np.max(y_truth)
N = X.shape[0]

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1


# In[4]:


# plot data points generated
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# ### Sigmoid function
# 
# 
# $\textrm{sigmoid}(\boldsymbol{W}^{\top} \boldsymbol{x} + w_{0}) = \dfrac{1}{1 + \exp\left[-(\boldsymbol{W}^{\top} \boldsymbol{x} + w_{0})\right]}$

# ### Error Function
# given as :
# 
# $\textrm{Error} =\begin{align*} 0.5\sum\limits_{i = 1}^{N} \sum\limits_{c = 1}^{K} \left[ (y_{ic}-\widehat{y}_{ic})^2 \right]\end{align*}$
# 
# Then:
# \begin{align*}
# \dfrac{\partial \textrm{Error}}{\partial \boldsymbol{w}_{c}} &= -\sum\limits_{i = 1}^{N} (y_{ic} - \widehat{y}_{ic})\widehat{y}_{ic}(1 - \widehat{y}_{ic})\boldsymbol{x}_{i} \\
# \dfrac{\partial \textrm{Error}}{\partial w_{c0}} &= -\sum\limits_{i = 1}^{N} (y_{ic} - \widehat{y}_{ic}) 
# \widehat{y}_{ic}(1 - \widehat{y}_{ic})\end{align*}

# In[5]:


def sigmoid(X, W, w_0):
    return(np.asarray([1 / (1 + np.exp(-(np.matmul(X, W[:,c]) + w_0[:,c])))for c in range (K)])).transpose()


def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([ -np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1)*
                               np.repeat((Y_predicted[:,c])[:, None], X.shape[1], axis = 1)*
                               np.repeat((1 - Y_predicted[:,c])[:, None], X.shape[1], axis = 1)*X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum((Y_truth - Y_predicted)*(Y_predicted)*(1 - Y_predicted), axis = 0))### Sigmoid function


# In[6]:


#set and initialize parameters 
eta = 0.01
epsilon = 0.001

W = np.random.uniform(low = -0.01, high = 0.01, size = (X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# In[7]:


# learn W and w0 using gradient descent
iteration = 1
objective_values = []

while 1:
    Y_predicted = sigmoid(X, W, w0)

    objective_values = np.append(objective_values, np.sum(0.5*(Y_truth -Y_predicted)**2))

    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(W)
print(w0)


# In[8]:


#draw objective function values throughout the iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[9]:


#predict y values
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
#calculate the confusion matrix
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[10]:


#create grids as in the lab sessions
x1_interval = np.linspace(-6, +6, 901)
x2_interval = np.linspace(-6, +6, 901)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

#create 0 matrix for initializing discriminant values
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

#for each class c determine the discriminant values
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]
#got partitions for each class
A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]

#for indicies where a class value is less then both of them ignore 
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C
#plot data
plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
#plot misclassified data points
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
#plot decision boundaries
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[ ]:




