import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split



#######################################
####Q2.1: Normalization




def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    shape = train.shape
    b = np.min(train,axis = 0)
    a = np.max(train,axis = 0) - b
    
    return ((train - b)/ a , (test - b)/a)
    

########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    m = X.shape[0] *1.0
    theta = np.array([theta]).T
    y = np.array([y]).T
    return (1/(2*m) *  np.linalg.norm(np.dot(X,theta) - y)**2)
    


########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    m = X.shape[0] *1.0
    theta = np.array([theta]).T
    y = np.array([y]).T
    
    return (1/m * np.dot(X.T,(np.dot(X,theta) - y)) )[:,0]

       
        
###########################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    
    y = np.array([y]).T
    approx_matrix = np.identity(num_features)
    approx_grad = np.apply_along_axis(lambda t: (compute_square_loss(X,y,theta+ epsilon*t) - compute_square_loss(X,y,theta- epsilon*t))/(2*epsilon) ,0,approx_matrix)
    
    checker = np.linalg.norm(true_gradient - approx_grad)
    
    
    
    return checker < tolerance
    
      
    
#################################################
###Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func,
                             gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters.
    And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss,
    and gradient_func = compute_square_loss_gradient
    """

    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    
    y = np.array([y]).T
    approx_matrix = np.identity(num_features)
    approx_grad = np.apply_along_axis(lambda t: (objective_func(X,y,theta+ epsilon*t)
                                                 - objective_func(X,y,theta- epsilon*t))/(2*epsilon) ,0,approx_matrix)
    
    checker = np.linalg.norm(true_gradient - approx_grad)
    
    
    return checker < tolerance
 

    


####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    
    theta_hist[0,:] = theta
    loss_hist[0] = compute_square_loss(X,y,theta)
    
    if not check_gradient:
        for i in list(xrange(1,num_iter+1)):
            theta = theta - compute_square_loss_gradient(X,y,theta)*alpha
            theta_hist[i,:] = theta 
            loss_hist[i] = compute_square_loss(X,y,theta)
    else:
        for i in list(xrange(1,num_iter+1)):
            step_checker = grad_checker(X,y,theta,epsilon=0.01, tolerance=1e-4)
            if step_checker:
                theta = theta - compute_square_loss_gradient(X,y,theta)*alpha
                theta_hist[i,:] = theta 
                loss_hist[i] = compute_square_loss(X,y,theta)        
            else:
                print('error, please check gradient function')
                return 
            
    return (theta_hist,loss_hist)

def step_checker(X, y, alpha = [0.1], num_iter=1000):
    num_alpha = len(alpha)
    loss_hist  = np.zeros((num_alpha, 1001))
    for i in list(xrange(num_alpha)):
        loss_hist[i,:] = batch_grad_descent(X,y,alpha[i],num_iter=1000)[1]
    
    return loss_hist


####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
def batch_line_search_grad_descent(X,y,num_iter=1000,tau = 0.5,c = 0.5,alpha_0 = 0.5):
    
    """
    use backtracking line search to minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        num_iter - number of iterations to run 
        tau,c,alpha_0 - float number, the parameters used for step search 
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
        time_list - the time used to each the step size for each iteration, 1D numpy array
    """
    
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    
    theta_hist[0,:] = theta
    loss_hist[0] = compute_square_loss(X,y,theta)
    time_list = np.zeros(num_iter)

    for i in list(xrange(1,num_iter+1)):
        
        alpha = alpha_0 
        p = -compute_square_loss_gradient(X,y,theta)
        t = 1.0 *c * np.linalg.norm(p)**2
        temp1 = compute_square_loss(X,y,theta)  #find current loss
        current  = time.time() 
        while 1: 
            temp2 = compute_square_loss(X,y,theta+ alpha *p) # compute loss at current step
            if   temp2 >  temp1 - alpha * t: 
                alpha = tau* alpha  #shrink alpha
                continue
            else:
                time_list[i-1] = time.time() -current
                break
        
        theta = theta+ alpha *p
        theta_hist[i,:] = theta 
        loss_hist[i] = temp2
                  
            
    return (theta_hist,loss_hist,time_list)    

plt.close('all')
#temp = batch_line_search_grad_descent(X_train,y_train)
plt.subplot()
plt.ylim([0,20])
plt.ylabel('objective function')
plt.xlabel('step')

plt.plot(temp[1])   

plt.show()
plt.savefig('2_4_3.png')

current1  = datetime.datetime.now()
batch_line_search_grad_descent(X_train,y_train)
print('runing time for backtracking line search GD' + str(datetime.datetime.now() -current1) )

current2  = datetime.datetime.now()
batch_grad_descent(X_train,y_train)
print('runing time for Batch GD ' + str(datetime.datetime.now() -current2) )


current3 = np.mean(batch_line_search_grad_descent(X_train,y_train)[2])
print('average extra runing time for step search: ' + str(current3) )

current4  = time.time()
compute_square_loss_gradient(X_train,y_train,theta)
print('runing time for computing gradient ' + str(time.time() - current4) )


###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    return compute_square_loss_gradient(X,y,theta) + 2*lambda_reg*theta 

###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    
    theta_hist[0,:] = theta
    loss_hist[0] = compute_square_loss(X,y,theta) + lambda_reg*np.linalg.norm(theta)**2
    
    for i in list(xrange(1,num_iter+1)):
        theta = theta - compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)*alpha
        theta_hist[i,:] = theta 
        loss_hist[i] = compute_square_loss(X,y,theta) + lambda_reg*np.linalg.norm(theta)**2
            
    return (theta_hist,loss_hist)    
    
    
    
    
    
#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss






lambda_=[10**-7,10**-5,10**-3,10**-1,1,10,100]

def Ridge_lambda_search(X_train = X_train, y_train = y_train,X_test= X_test, y_test = y_test, lambda_=[10**-7,10**-5,10**-3,10**-1,1,10,100]): 
    """
    checking the convergency for different regulization lambda
    returns the last element of loss_hist returned by regularized_grad_descent
    """
    
    res_training =  np.zeros(len(lambda_))
    res_testing = np.zeros(len(lambda_))
                    
    for i in list(xrange(len(lambda_))):
        temp = regularized_grad_descent(X_train,y_train,lambda_reg=lambda_[i])
        theta = temp[0][-1,:]
        res_testing[i] = compute_square_loss(X_test,y_test,theta)
        res_training[i] = compute_square_loss(X_train,y_train,theta)

    return (res_testing,res_training)

plt.close('all')
try_list = np.linspace(10**-7,10**-1,100)
t_2_5 = Ridge_lambda_search(X_train, y_train, X_test, y_test, lambda_=try_list)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0,3])
ax.set_xlabel('$log_{10}(\lambda)$')
ax.set_ylabel('loss')

plt.plot(np.log10(try_list),t_2_5[0],label = 'validation loss')
plt.plot(np.log10(try_list),t_2_5[1],label = 'training loss')
plt.legend()
plt.show()
plt.savefig('2_5_5.png')


lambda_opt=  try_list[np.where(t_2_5[0] ==np.min(t_2_5[0]))[0][0]]

##2.5.6

df = pd.read_csv('hw1-data.csv', delimiter=',')
X = df.values[:,:-1]
y = df.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

X_train, X_test = feature_normalization(X_train, X_test)

try_list = np.linspace(0,10,110)
res = np.ones(110)

for i,B in enumerate(try_list):
    data = (np.hstack((X_train, B*np.ones((X_train.shape[0], 1)))),np.hstack((X_test, B*np.ones((X_test.shape[0], 1)))))
    
    res[i] = Ridge_lambda_search(data[0],y_train,data[1],y_test,lambda_ = [1])[1][0]

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0,20])
ax.set_xlabel('B')
ax.set_ylabel('validation loss')

plt.plot(try_list,res)
plt.show()
plt.savefig('2_5_6.png')


current = time.time()
regularized_grad_descent(X_train,y_train,lambda_reg = lambda_opt)
print (time.time() - current)/1000



#############################################
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    t = 0
    
    for n in list(xrange(num_iter)):
        generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
        
        for i in list(xrange(0,num_instances)):
            t+=1.0
            num = generator[i]
            if alpha == "1/sqrt(t)":
                theta = theta - 1/np.sqrt(t) *( (np.inner(X[num,:], theta)-y[num]) *X[num,:] + 2*lambda_reg*theta)
            elif alpha  == "1/t":
                theta = theta - (1/(t)) * ( (np.inner(X[num,:], theta)-y[num]) *X[num,:] + 2*lambda_reg*theta)
            else:
                theta = theta - alpha * ((np.inner(X[num,:], theta)-y[num]) *X[num,:] + 2*lambda_reg*theta)
            
            theta_hist[n][i,:] = theta 
            loss_hist[n,i] = compute_square_loss(X,y,theta)+lambda_reg*np.linalg.norm(theta)**2
        
    return (theta_hist,loss_hist)    
    

################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

t_6_3 = np.array([0,0,0,0],dtype = object) 

for i,a in enumerate([0.0001,0.001,0.01,0.05]):
    t_6_3[i] = stochastic_grad_descent(X_train,y_train,alpha = a,lambda_reg = lambda_opt)[1]

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([1,50])
ax.set_xlabel('Epoch')
ax.set_ylabel('objective function value')


for i in [0,1,2,3]:
    plt.plot(list(range(1,1001)),t_6_3[i][:,-1],label = 'step size = ' + str([0.0001,0.001,0.01,0.05][i]))
    
plt.legend()
plt.show()
plt.savefig('2_6_3_1.png')

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([1,30])
ax.set_xlabel('Epoch')
ax.set_ylabel('objective function value')

plt.plot(list(range(1,1001)),stochastic_grad_descent(X_train,y_train,alpha = '1/t')[1][:,-1],label = r'$\eta_t = \frac{1}{t}$')
plt.plot(list(range(1,1001)),stochastic_grad_descent(X_train,y_train,alpha = '1/sqrt(t)')[1][:,-1],label = r'$\eta_t = \frac{1}{\sqrt{(t)}}$')

plt.legend(loc='upper right')
plt.show()
plt.savefig('2_6_3_2.png')


#2.6.4
def stochastic_grad_descent_opt(X, y,eta_0, lambda_reg=1, num_iter=1000):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    t = 0
    
    for n in list(xrange(num_iter)):
        generator = np.random.permutation(list(xrange(num_instances))) # define ramdom sampling sequence
        
        for i in list(xrange(0,num_instances)):
            t+=1.0
            num = generator[i]
    
            theta = theta - (eta_0/(1+eta_0*lambda_reg*t)) *( (np.inner(X[num,:], theta)-y[num]) *X[num,:] + 2*lambda_reg*theta)

            
            theta_hist[n][i,:] = theta 
            loss_hist[n,i] = compute_square_loss(X,y,theta)+lambda_reg*np.linalg.norm(theta)**2
        
    return (theta_hist,loss_hist) 


res = np.zeros(5,dtype = object)

try_list = [0.001,0.005,0.01,0.05,0.1]

for i,eta in enumerate(try_list):
    res[i] = stochastic_grad_descent_opt(X_train,y_train,eta,lambda_reg = lambda_opt)[1]


plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([1,30])
ax.set_xlabel('Epoch')
ax.set_ylabel('objective function value')

for i in list(xrange(5)):
    plt.plot(res[i][:,-1],label = r'$\eta_0$ = '+str(try_list[i]))

plt.legend()
plt.show()
plt.savefig('2_6_4.png')

current  = time.time()
stochastic_grad_descent(X_train,y_train,alpha = '1/t')
print('runing time for a single epoch of SGD ' + str((time.time() -current)/1000) + 'second')






def main():
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term

    #2.4.2
    plt.close('all')
    alpha = [0.5,0.1,0.05,0.01] 
    temp = step_checker(X_train,y_train,alpha,1000)
    plt.subplot()
    plt.ylim([0,20])
 
    plt.savefig('2_4_2.png')

    #2.6.3

    
    


if __name__ == "__main__":
    main()
