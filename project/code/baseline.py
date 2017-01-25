
import numpy as np
from scipy.sparse import *
from scipy import io
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
import csv

def save_sparse_csc(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])




# constructing data
coupon_list = np.array([i.strip().split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv')],dtype = object)[1:,-1]
user_list = np.load("/Users/cryan/Google Drive/1003_project/data/user.npy")[1:,:]
purchase_hist = np.array([i.strip().split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/coupon_detail_train.csv')],dtype = object)[1:,[0,4,5]]
coupon_test = np.array([i.strip().split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv')],dtype = object)[1:,-2:]
coupon_visit = np.array([i.strip().split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/coupon_vist_train.csv')],dtype = object)[1:,]

coupon_hash = defaultdict()
user_hash = defaultdict()

for i in xrange(len(coupon_list)):
    coupon_hash[coupon_list[i]] = i

for i in xrange(len(user_list)):
    user_hash[user_list[i,5]] = i

purchase_matrix= lil_matrix( (len(user_hash),len(coupon_hash)), dtype=int )
for i in purchase_hist:
    purchase_matrix[user_hash[i[1]],coupon_hash[i[2]]] = int(i[0])

purchase_matrix = purchase_matrix.tocsc()
#io.mmwrite("purchase_matrix.mtx", purchase_matrix)

#predicting
coupon_list_test = pd.read_csv("/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv",sep=',',header = False)['COUPON_ID_hash']
baselin_res4 = np.load("/Users/cryan/Google Drive/1003_project/data/baseline_result_5.npy")
out4 = np.array(map(lambda x:[coupon_list_test[i] for i in x] ,baselin_res4))

out4 = [[user_list[i]]+ list(out4[i]) for i in range(len(baselin_res4))]
out4 = np.array(out4,dtype = object)

test4 = [[out4[i][0]] +[' '.join(out4[i][1:])] for i in range(len(baselin_res4))]
test4 = np.array(test4,dtype= object)

#test4 = np.vstack((np.array(['USER_ID_hash','PURCHASED_COUPONS']) , test4))
#np.savetxt('output4.csv',out4,delimiter=',')
with open('output5.csv','wb') as f:
    csv.writer(f).writerows(test4)

# logistic regression on 100 centroid


coupon_matrix_train = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_train.npy')
coupon_matrix_test = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_test.npy')
cluster_list = io.mmread('/Users/cryan/Google Drive/1003_project/data/data_logistic_regression/cluster_list_2.mtx').toarray()
user_centroid_weight = np.load('/Users/cryan/Google Drive/1003_project/data/data_logistic_regression/user_centroid_weight.npy')
user_centroid = np.load('/Users/cryan/Google Drive/1003_project/data/data_logistic_regression/user_centroid_2.npy')

model_list = np.zeros(1000, dtype= object)
predict_lik = np.zeros((1000,coupon_matrix_test.shape[0]))


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


for i in xrange(1000):
    print('stage'+ str(i) + '\n')
    weight = (cluster_list[i,:]+1 )/np.sum(cluster_list[i,:]+1 )
    labels = 1 - (cluster_list[i,:] == 0)
    if np.sum(labels) != 0:
        clf = LogisticRegression(C=10000, penalty='l2',solver= 'sag')
        clf.fit(coupon_matrix_train,labels,sample_weight= weight)
        predict_lik[i,:] = np.array(map(lambda x: clf.predict_proba(x)[0][1],coupon_matrix_test))
        model_list[i] = clf
    else:
        model_list[i] = None
        predict_lik[i,:] = np.zeros(310)


res = np.dot(user_centroid_weight,predict_lik)
#res = np.argsort(res,axis = -1)[:,-10:]

predict_lik = res
temp =np.argsort(predict_lik,axis = -1)[:,-10:][:,::-1]

prediction = np.zeros((22873,2),dtype=object)

for i,j in enumerate(user_centroid):
    prediction[i,0] = user_list[i]
    prediction[i,1] = ' '.join(map(lambda x:coupon_test[x] ,temp[j,:]))


prediction = prediction[prediction[:,0].argsort()]
prediction = np.vstack((np.array(['USER_ID_hash','PURCHASED_COUPONS']),prediction))
with open('output6.csv','wb') as f:
    csv.writer(f).writerows(prediction)


#psudo user location:
user_list = np.load("/Users/cryan/Google Drive/1003_project/data/user.npy")[1:,:]
user_list = np.hstack((user_list,np.zeros((user_list.shape[0],2))))
for i,j in enumerate(user_list):
    user[i,-1] = []
    if user_list[i,-2] != '':
        user_list[i,-2] = (float(user_list[i,-2]) - mu_long)/std_long
        user_list[i,-3] = (float(user_list[i,-3]) - mu_lat)/std_lat
        purchased = purchase_hist[purchase_hist[:,1] == j[5]][:,[0,2]]



coupon_full = np.array([i.strip().split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv')],dtype = object)[1:,]

def get_aviliable(time):
    res = []
    for i,j in enumerate(coupon_full):
        if (datetime.strptime(time,"%Y-%m-%d %H:%M:%S") > j[5]) and (datetime.strptime(time,"%Y-%m-%d %H:%M:%S") < j[6]):
            res.append(j[-1])
    return res

def user_visit(id):
    visit = defaultdict()
    visit_list = []
    for i,j in  enumerate(coupon_visit):
        if j[-3] == id:
            visit.append()





