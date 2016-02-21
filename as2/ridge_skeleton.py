import numpy as np
import matplotlib.pyplot as plt
import time
from hw1 import *
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge


#1.1.1
X = np.random.rand(150,75)
#1.1.2
theta_true = 20*numpy.random.randint(2,size = 10)-10
theta_true = np.array([np.hstack((theta_true,np.zeros(65)))]).T
#1.1.3
noise = np.array([0.1*np.random.randn(150)]).T
y = np.dot(X,theta_true) + noise
#1.1.4

X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size =50, random_state=10)
X_train,X_validation,y_train,y_validation = train_test_split(X_train_, y_train_, test_size =20, random_state=11)

#1.2.1
(N,D) = X_train.shape

w = numpy.random.rand(D,1)

def ridge(Lambda):
  def ridge_obj(theta):
    return ((numpy.linalg.norm(numpy.dot(X_train,theta) - y_train))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
  return ridge_obj

def compute_loss(Lambda, theta):
  return ((numpy.linalg.norm(numpy.dot(X_validation,theta) - y_validation))**2)/(2*N)


for i in range(-7,6):
  Lambda = 10**i;
  w_opt = minimize(ridge(Lambda), w)
  hist.append(compute_loss(Lambda, w_opt.x))
  print Lambda, hist[-1]
  
#lambda_opt =  list(range(-,7))[np.where(hist  ==np.nanmin(hist))[0][0]]

temp = Ridge_lambda_search(X_train,y_train[:,0],X_validation,y_validation[:,0],lambda_ = np.power(10, np.linspace(-7,6,100)))[0]

lambda_opt = np.power(10, np.linspace(-7,6,100))[np.where(temp  ==np.nanmin(temp))[0][0]]


beta_= regularized_grad_descent(X_train,y_train[:,0],lambda_reg = lambda_opt )[0][-1,:]
np.sum(theta_true[10:,:] != np.array([beta_]).T[10:])







####################################
###Q2.1

def LassoShooting(X, y, lambda_ = 0.1,epsilon = 0.0001,n_iter = 1000,beta_init = np.zeros((X.shape[1],1)) ):
  num_instances, num_features = X.shape      
  beta = beta_init
  t = 0
  converged = False
  Loss = np.linalg.norm(np.dot(X,beta) - y)**2 *(1.0/(2*num_instances))
  while (not converged and t <=n_iter ):
    beta_start = beta
    for j in range(num_features):
      aj = 0
      cj = 0
      for i in range(num_instances):
        aj += 2*X[i,j]**2
        cj += 2*X[i,j] *(y[i][0] - np.inner(beta_start[:,0],X[i,:])+beta_start[j][0]*X[i,j])
      
      beta[j] = np.sign(aj)* (lambda x: np.sign(x[0])* ((abs(x[0]) - x[1]) if (abs(x[0]) - x[1])>0 else 0))([cj/aj,lambda_/aj])
    t+=1
    converged = abs(Loss - np.linalg.norm(np.dot(X,beta) - y)**2*(1.0/(2*num_instances))) < epsilon
    Loss = np.linalg.norm(np.dot(X,beta) - y)**2*(1.0/(2*num_instances))
  return (beta,Loss)
  
loss_hist = np.zeros(10)  
for i,reg in enumerate(np.power(10,np.linspace(-5,4,10))):
  beta = LassoShooting(X_train,y_train,lambda_ = reg)[0]
  loss_hist[i] = np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2 *(1.0/(2*num_instances))
 
 
try_list =  np.power(10,np.linspace(0,2,30)) 
loss_hist = np.zeros(try_list.shape[0])
v_hist = np.zeros(try_list.shape[0])

time1 = time.time()
for i,reg in enumerate(try_list):
  beta = LassoShooting(X_train,y_train,lambda_ = reg)[0]
  loss_hist[i] = np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2*(1.0/(2*num_instances))
  v_hist[i] = np.linalg.norm(np.dot(X_train,beta) - y_train)**2*(1.0/(2*num_instances))
time_hist_1 = time.time()-time1 
 
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0,1])
ax.set_xlabel('$log_{10}(\lambda)$')
ax.set_ylabel('Validation loss')

plt.plot(np.log10(try_list), loss_hist,label = 'validation error')
plt.plot(np.log10(try_list), v_hist,label = 'training error')
plt.show()
plt.legend(loc = 2)
plt.savefig('2_1.png')

lambda_opt =  try_list[np.where(loss_hist == np.min(loss_hist))[0][0]]
beta_opt = LassoShooting(X_train,y_train,lambda_ = lambda_opt)[0]
test_error = np.linalg.norm(np.dot(X_test,beta_opt) - y_test)**2*(1.0/(2*num_instances))

###########2.1.2
np.sum(theta_true[10:,:] != beta_opt[10:,:])

###########2.1.3
loss_hist_3 = np.zeros(try_list.shape[0])
pre = np.zeros((75,1))

time1 = time.time()
for i,reg in enumerate(try_list):
  beta = LassoShooting(X_train,y_train,lambda_ = reg,beta_init =pre )[0]
  pre,loss_hist[i] = beta, np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2*(1.0/(2*num_instances))
time_hist_2 = time.time()-time1


loss_hist = np.zeros(10)  
for i,reg in enumerate(np.power(10,np.linspace(-5,4,10))):
  beta = LassoShooting(X_train,y_train,lambda_ = reg)[0]
  loss_hist[i] = np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2*(1.0/(2*num_instances))



###########2.1.4
def LassoShooting_mat(X, y, lambda_ = 0.1,epsilon = 0.0001,n_iter = 1000,beta_init = np.zeros((X.shape[1],1)) ):
  num_instances, num_features = X.shape      
  beta = beta_init
  t = 0
  converged = False
  XX2 = np.dot(X.T,X)*2;
  Xy2 = np.dot(X.T,y)*2
  Loss = np.linalg.norm(np.dot(X,beta) - y)**2*(1.0/(2*num_instances))
  while (not converged and t <=n_iter ):
    beta_start = beta
    for j in range(num_features):
      aj = XX2[j,j]
      cj = (Xy2[j] - np.dot(XX2[j,:],beta_start) + XX2[j,j]*beta_start[j])[0]
      
      beta[j] = np.sign(aj)* (lambda x: np.sign(x[0])* ((abs(x[0]) - x[1]) if (abs(x[0]) - x[1])>0 else 0))([cj/aj,lambda_/aj])
    t+=1
    converged = abs(Loss - np.linalg.norm(np.dot(X,beta) - y)**2*(1.0/(2*num_instances))) < epsilon
    Loss = np.linalg.norm(np.dot(X,beta) - y)**2*(1.0/(2*num_instances))
  return (beta,Loss)
 
 
pre = np.zeros((75,1))
time1 = time.time()
for i,reg in enumerate(try_list):
  beta = LassoShooting_mat(X_train,y_train,lambda_ = reg,beta_init =pre )[0]
  pre,loss_hist[i] = beta, np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2*(1.0/(2*num_instances))
time_hist_2 = time.time()-time1



##########2.5.1
def projected_SGD(X, y, lambda_ = 0.1,alpha = 0.01, num_iter = 1000,theta_init = np.zeros((X.shape[1],1))):
  num_instances, num_features = X.shape[0], X.shape[1]
  theta_p = theta_init
  theta_n = theta_init
  theta = theta_p - theta_n
  
  for n in list(xrange(num_iter)):
    generator = np.random.permutation(list(xrange(num_instances))) 
    for i in list(xrange(0,num_instances)):

      num = generator[i]
      theta_p = theta_p - 0.01*(np.inner(X[num,:],theta.T) - y[num])* np.array([X[num,:]]).T +lambda_
      theta_n = theta_n +   0.01*(np.inner(X[num,:],theta.T) - y[num])* np.array([X[num,:]]).T +lambda_
      
      theta_p = (theta_p>0)*theta_p
      tehta_n = (theta_n<0) *theta_n
      theta = theta_p - theta_n
      
  
  loss_hist = np.linalg.norm(np.dot(X,theta)-y)**2 *(1.0/(2*num_instances))
      
  return (theta,loss_hist)


try_list =  np.power(10,np.linspace(-1,1,30)) 
loss_hist = np.zeros(try_list.shape[0])
loss_hist_2 = np.zeros(try_list.shape[0])

for i,reg in enumerate(try_list):
  beta = LassoShooting_mat(X_train,y_train,lambda_ = reg)[0]
  loss_hist[i] = np.linalg.norm(np.dot(X_validation,beta) - y_validation)**2*(1.0/(2*num_instances))
  beta_2 = projected_SGD(X_train,y_train,lambda_ = reg)[0]
  loss_hist_2[i] = np.linalg.norm(np.dot(X_validation,beta_2) - y_validation)**2*(1.0/(2*num_instances))
  
  
   
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0,0.3])
ax.set_xlabel('$log_{10}(\lambda)$')
ax.set_ylabel('Validation loss')

plt.plot(np.log10(try_list), loss_hist ,label = 'Shooting Method')
plt.plot(np.log10(try_list), loss_hist_2 ,label = 'Projected SGD')
plt.legend()
plt.show()
plt.savefig('5_1.png')


lambda_opt =  try_list[np.where(loss_hist_2 == np.min(loss_hist_2))[0][0]]
beta_opt = projected_SGD(X_train,y_train,lambda_ = lambda_opt)[0]
