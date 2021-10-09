#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[24]:


def predict(w,b,X):
    '''
    function to predict the y_hat
    i.e. y_hat = X*w + b
    '''
    pred = X@w + b
    return pred

def fit(y, y_hat, X, w, b, iteration=2, lr=0.01):
    '''
    function to fit the linear regression model in the given dataset
    gradient descent has been used to update the w & b parameters
    '''
    grad_w = np.empty((X.shape[1],1))
    for num_iter in range(iteration):
        for i in range(X.shape[1]):
            grad_w[i] = np.sum(2*np.multiply((y-y_hat),X[:,i].reshape(-1,1)))
        w = w - lr*grad_w
        grad_b = np.sum(2*(y*y_hat))
        b = b - lr*grad_b
    return w,b

def calc_R2(y,y_hat):
    '''
    function to calculate the R2 score
    R2 = [explained variance / total variance] = [1-(RSS/TSS)]
    RSS = difference between actuals and predictions
    TSS = difference between actuals and average
    '''
    RSS = np.sum(y-y_hat)
    TSS = np.sum(y-np.mean(y_hat))
    return 1-(RSS-TSS)
                               
def main():
    X = np.random.rand(10,2)
    y = np.random.randint(5,size=10).reshape(-1,1)
    w = np.random.rand(X.shape[1],1)
    b = np.random.rand()
    y_pred = predict(w,b,X)
    w,b = fit(y,y_pred,X,w,b)  
    print('w=',w)
    print('b=',b) 
    R2 = calc_R2(y,predict(w,b,X))
    print('R2_score=',R2)
    
    

                                         


# In[25]:


if __name__=='__main__':
    main()

