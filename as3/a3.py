from util import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import numpy as np
import pickle
import random
import time



def mapper(data):
    '''
    function for map reduce that chop dictionary into 5 dicts

    Args:
        data: the input dict

    Returns: tuple of 5 dict
    '''
    diff = len(data.keys())//5
    r1 = dict(data.items()[:diff])
    r2 = dict(data.items()[diff:2*diff])
    r3 = dict(data.items()[2*diff:3*diff])
    r4 = dict(data.items()[3*diff:4*diff])
    r5 = dict(data.items()[4*diff:])
    return (r1,r2,r3,r4,r5)


def scale(x,c):
    '''
    scale dicitionary by constant c

    Args:
        x:
        c:

    Returns:

    '''
    if x == Counter():
        return x
    else:
        temp = np.array(x.items(),dtype= object)
        temp[:,1] = temp[:,1]*c
        return dict(temp)



def pegasos_svm_sgd(X,y,lambda_ = 10,n_ite = 1,print_time = False):
    '''
    pegasos svm with pure sgd approach

    Args:
        X: Train data
        y: Train lable
        lambda_: regulization
        n_ite: max iterations
        print_time: whether count the operation time

    Returns: sparse representaion of the weight

    '''
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
                increment(w,- eta*lambda_,w)
                increment(w,eta*y[i],X[i])
            else:
                increment(w,- eta*lambda_,w)
        n+=1
    if print_time:
        print( time.time() -time_ )
    return w


def pegasos_svm_sgd_2(X,y,lambda_ = 1,n_ite = 10,counter = False,print_time = False):
    '''
    updated pegasos svm with pure sgd approach

    Args:
        X: Train data
        y: Train lable
        lambda_: regulization
        n_ite: max iterations
        counter: whether count the # of nondifferentiable case
        print_time: whether count the operation time

    Returns: sparse representaion of the weight

    '''
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    num_instances = X.shape[0]
    t = 1.0
    n = 0
    W = Counter()
    s=1.0
    count = 0.0
    time_ = time.time()
    while n < n_ite:
        generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
        for i in generator:
            t+=1
            eta = 1/(t*lambda_)
            s = (1 - eta*lambda_)*s
            temp = dotProduct(W,X[i])
            if temp ==0 and counter==True:
                count+=1.0
            if temp*y[i] <1/s:
                increment(W,1/s *eta *y[i], X[i])

        n+=1
    if print_time:
        print( "runtime = "+str(time.time() -time_ ))
    if counter:
        print count/(num_instances*n_ite)
    return scale(W,s)


def loss_svm(X,y,w):
    '''
    loss function for svm

    Args:
        X: testing data
        y: true lables
        w: weight

    Returns: loss

    '''
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    return np.mean(map(lambda t: np.max(np.array((t,0))), 1- np.array(map(lambda t: dotProduct(w,t),X))*y))


def loss_0_1(X,y,w):
    '''
    0_1 loss function for svm

    Args:
        X: testing data
        y: true lables
        w: weight

    Returns: loss

    '''
    X = np.array(map(lambda a: tokenlizer(a) , X),dtype = object)
    return np.mean(map(lambda t: 0 if t>=0 else 1, np.array(map(lambda t: dotProduct(w,t),X))*y))


def list_feature(X,w,n):
    '''
    show n heavily features

    Args:
        X: input data
        w: weightes
        n: number of features to display

    Returns: r1:word, r2 number cotained in input, r3 weight, r4 contribution

    '''
    temp = Counter()
    for i,j in w.items():
        temp[i] = abs(j*X[i])
    res = np.zeros((4,n),dtype=object)
    for k,i in enumerate(temp.most_common(n)):
        res[:,k] = i[0],X[i[0]],w[i[0]],abs(X[i[0]] *w[i[0]])
    return res

def remover(x,sw):
    res = []
    for i in x:
        if i not in sw:
            res.append(i)
    return res

def remover_2(x,sw):
    res = []
    n = len(x)-1
    i=0
    negative = ['not',"aren't","isn't","doesn't","don't","didn't","hasn't","haven't","shoulden't","wouldn't","won't"]
    while i <= n:
        cur = x[i]
        if (cur not in sw) and (cur != 'not'):
            res.append(cur)
        elif cur == 'not' and i !=n:
            res.append(cur+'_'+x[i+1])
            i+=1
        i+=1
    return res
if __name__ == '__main__':

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("Y_train.npy")
    y_test  = np.load("Y_test.npy")


    w1 = pegasos_svm_sgd(X_train,y_train,1,1,print_time=True)  # 35s
    l1 = loss_0_1(X_train,y_train,w1)

    w2 = pegasos_svm_sgd_2(X_train,y_train,1,1) # 0.38 s
    l2 = loss_0_1(X_train,y_train,w2)

    #####6.6
    try_list = np.power(10.0,list(range(-8,5)))
    loss_list = np.zeros(13)
    for i,j in enumerate(try_list):
        w = pegasos_svm_sgd_2(X_train,y_train,lambda_ = j,n_ite = 10)
        loss_list[i] = loss_0_1(X_test,y_test,w)


    ###6.7
    try_list_2 = np.power(10.0,np.linspace(-5,-4,20))
    loss_list_2 = np.zeros(20)
    for i,j in enumerate(try_list_2):
        w = pegasos_svm_sgd_2(X_train,y_train,lambda_ = j,n_ite = 40)
        loss_list_2[i] = loss_0_1(X_test,y_test,w)

    lambda_opt = try_list_2[np.where(loss_list_2 == min(loss_list_2))[0][0]]
    w_opt = pegasos_svm_sgd_2(X_train,y_train,lambda_ = lambda_opt,n_ite = 40,print_time= True)

    X = np.array(map(lambda a: tokenlizer(a) , X_test),dtype = object)
    y = y_test
    score = np.abs(np.array(map(lambda t: dotProduct(w_opt,t),X)))
    error =np.array( map(lambda t: 0 if t>0 else 1, np.array(map(lambda t: dotProduct(w,t),X))*y))
    score_list = np.array([score,error]).T
    score_list_sorted = score_list[score_list[:,0].argsort()]
    temp =np.linspace(np.min(score),np.max(score),7)
    index = map(lambda i:np.where(score_list_sorted[:,0] <= i )[0][-1] ,temp)
    range_by_index = score_list_sorted[index][:,0]
    error_by_index = map(lambda i:np.mean(score_list_sorted[:,1][index[i]:index[i+1]]),[0,1,2,3,4,5])
    new_xstick = map(lambda i: str(range_by_index[i])[0:5]+ ' to ' +  str(range_by_index[i+1])[0:5] ,[0,1,2,3,4,5])
    plt.close('all')
    fig =plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar([3,6,9,12,15,18],error_by_index)
    sti_loc = np.array([3,6,9,12,15,18])+0.5
    ax1.set_xticks(sti_loc)
    ax1.set_xticklabels(new_xstick,size = 'x-small')
    ax1.set_xlim([1,19])
    ax1.set_xlabel("magnitude of the absolute score")
    ax1.set_ylabel('percentage error')
    plt.show()
    plt.savefig('6_7.png')

    ###6.8
    pegasos_svm_sgd_2(X_train,y_train,lambda_ = lambda_opt,n_ite = 40,counter = True)



    #7
    error_list = np.where(np.array(prediction) ==1)[0][:4]
    txt_error = X[error_list]

    res = []
    for i in txt_error:
        res.append(list_feature(i,w_opt,5))
    res = np.array(res)


    ################8.1

    from nltk.corpus import stopwords
    stopwords = stopwords.words("english")

    X_train_new =np.array(map(lambda x:remover(x,sw = stopwords),X_train))


    loss_list_3 = np.zeros(20)
    for i,j in enumerate(try_list_2):
        w = pegasos_svm_sgd_2(X_train_new,y_train,lambda_= j,n_ite=20)
        loss_list_3[i] = loss_0_1(X_test,y_test,w)


    ##################8.2
    X_train_2 = np.array(map(lambda x:remover_2(x,sw = stopwords),X_train))
    loss_list_4 = np.zeros(20)
    for i,j in enumerate(try_list_2):
        w = pegasos_svm_sgd_2(X_train_2,y_train,lambda_= j,n_ite=40)
        loss_list_4[i] = loss_0_1(X_test,y_test,w)

    ###8.3
    def list_to_str(x):
        res = ""
        for i in x:
            res+= (i + ' ')

        return res

    from  sklearn.feature_extraction.text import CountVectorizer

    str_list= map(lambda x:list_to_str(x), X_train)
    str_list.extend(map(lambda x:list_to_str(x), X_test))
    data_mat = CountVectorizer(analyzer= 'word',stop_words = 'english')
    data_mat = data_mat.fit_transform(str_list)
    data_mat = data_mat.toarray()

    data_mat_train = data_mat[:1500,:]
    data_mat_test = data_mat[1500:,:]

    def g_ker(x,w,s):
        return np.exp(np.linalg.norm(w-x)**2/(-2*s^2))

    Kernel = np.identity(1500)
    for i in range(1500):
        for j in range(i+1,1500):
            temp = g_ker(data_mat_train[i],data_mat_train[j],1)
            Kernel[i,j] = temp
            Kernel[j,i] = temp


    def Kernel_svm(K,y,lambda_=1,n_ite=10):
        num_instances =K.shape[0]
        w = np.zeros(n_feature)
        n=0.0
        t=1.0
        while n < n_ite:
            generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
            for i in generator:
                t+=1
                eta = 1/(t)
                temp = np.dot(w,K)
                if 1- temp[i]*y[i] <=0:
                    w = w - eta * (lambda_ *temp)
                else:
                    w = w - eta *(lambda_*temp - y[i]*K[i,:] )
            n+=1

        return w

    def error_kernel(K,x,x_test,y,w,s):
        temp =np.array(map(lambda i: np.sum(np.array(map(lambda j:g_ker(x[j],x_test[i],s),list(xrange(1500)))) * w),list(xrange(500))))*y
        return np.mean(np.array(map(lambda t: 1 if t<0 else 0 ,temp)))


    t = Kernel_svm(Kernel,y_train,lambda_ = 0.001,n_ite=20)

    try_list = np.power(10.0,list(range(-6,0)))
    loss_list = np.zeros(6)
    for i,j in enumerate(try_list):
        w = Kernel_svm(Kernel,y_train,lambda_ = j,n_ite = 20)
        loss_list[i] = error_kernel(Kernel,data_mat_train,data_mat_test,y_test,w,s)

    from sklearn.svm import SVC
    clf = SVC(kernel ='rbf',C = 0.001 )
    clf.fit(data_mat_train,y_train)
    clf.predict(data_mat_test)



