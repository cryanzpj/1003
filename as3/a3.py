
from util import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing
import os
import numpy as np
import pickle
import random
import time

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("Y_train.npy")
y_test  = np.load("Y_test.npy")


def mapper(data):
    diff = len(data.keys())//5
    r1 = dict(data.items()[:diff])
    r2 = dict(data.items()[diff:2*diff])
    r3 = dict(data.items()[2*diff:3*diff])
    r4 = dict(data.items()[3*diff:4*diff])
    r5 = dict(data.items()[4*diff:])
    return (r1,r2,r3,r4,r5)

def scale(x,c):
    if x == Counter():
        return x
    else:
        temp = np.array(x.items(),dtype= object)
        temp[:,1] = temp[:,1]*c
        return dict(temp)



def pegasos_svm_sgd(X,y,lambda_ = 10,n_ite = 1):
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    num_instances = X.shape[0]
    t = 0.0
    n = 0
    w = Counter()
    time_ = time.time()
    while n < n_ite:
        generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
        for i in generator:
            t+=1
            eta = 1/(t*lambda_)
            if dotProduct(w,X[i])*y[i] <1:
                #w = scale(w,- eta*lambda_)
                increment(w,- eta*lambda_,w)
                increment(w,eta*y[i],X[i])
            else:
                #w = scale(w,- eta*lambda_)
                increment(w,- eta*lambda_,w)
        n+=1
    print( time.time() -time_ )
    return w

def pegasos_svm_sgd_2(X,y,lambda_ = 10,n_ite = 1):
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    num_instances = X.shape[0]
    t = 1.0
    n = 0
    W = Counter()
    s=1.0
    time_ = time.time()
    while n < n_ite:
        generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
        for i in generator:
            t+=1
            eta = 1/(t*lambda_)
            s = (1 - eta*lambda_)*s
            if dotProduct(W,X[i])*y[i] <1/s:
                increment(W,1/s *eta *y[i], X[i])
        n+=1
    print( time.time() -time_ )
    return scale(W,s)

def loss(X,y,w):
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    return np.mean(map(lambda t: np.max(np.array((t,0))), 1- np.array(map(lambda t: dotProduct(w,t),X))*y))



t = pegasos_svm_sgd_2(X_train,y_train,1,1)
loss(X_train,y_train,t)
