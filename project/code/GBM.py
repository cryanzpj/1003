import numpy as np
import csv
import pandas as pd
import os
import sys
from scipy.sparse import *
from scipy import io
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from datetime import datetime
from datetime import timedelta


os.chdir("/Users/cryan/Google Drive/1003_project/code")


coupon_list_train = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv')
coupon_list_test = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv')
user_list = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/user_list.csv')
coupon_purchases_train = pd.read_csv("/Users/cryan/Google Drive/1003_project/data/coupon_detail_train.csv")


##featurlize data

features = ['DISPFROM','COUPON_ID_hash','GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE', 'large_area_name', 'ken_name', 'small_area_name']
categoricals = ['GENRE_NAME','large_area_name', 'ken_name', 'small_area_name']
coupon_list_train = coupon_list_train[features]
coupon_list_train['type'] = np.repeat('train',coupon_list_train.shape[0])
coupon_list_test = coupon_list_test[features]
coupon_list_test['type'] = np.repeat('test',coupon_list_test.shape[0])
coupon_list_full = coupon_list_train.append(coupon_list_test)
coupon_list_full['DISCOUNT_PRICE'] = 1 / np.log10(coupon_list_full['DISCOUNT_PRICE'])
coupon_list_full['PRICE_RATE'] = (coupon_list_full['PRICE_RATE'] / 100) ** 2
categoricals_features = coupon_list_full[categoricals]
coupon_list_full  = pd.concat([coupon_list_full, pd.get_dummies(categoricals_features, dummy_na=False)], axis=1)
coupon_list_full = coupon_list_full.drop(categoricals,axis=1)

coupon_list_train = coupon_list_full.iloc[(coupon_list_full['type'] == 'train').values].drop('type',1)
coupon_list_test = coupon_list_full.iloc[(coupon_list_full['type'] == 'test').values].drop('type',1)

val_inx = np.array(list(map(lambda x: (x <= datetime(2012, 6, 23)) and  (x > datetime(2012, 4, 19)),pd.to_datetime(coupon_list_train['DISPFROM']))))

coupon_val = coupon_list_train.iloc[val_inx]
coupon_train = coupon_list_train.iloc[val_inx == False]

coupon_byweek = np.zeros((43,3),dtype= object)
for i in range(43):
    coupon_byweek[i] = [datetime(2011, 6, 23)+ i*timedelta(7),datetime(2011, 6, 23)+ (i+1)*timedelta(7),pd.DataFrame(columns = coupon_list_train.columns.delete(0))]
coupon_train['DISPFROM'] = pd.to_datetime(coupon_train['DISPFROM'])


for j,i in coupon_train.iterrows():
    row = np.argmax(map(lambda x: i['DISPFROM'] > x[0] and i['DISPFROM'] <= x[1], coupon_byweek[:,:2]))
    coupon_byweek[row][2] = coupon_byweek[row][2].append(i.drop('DISPFROM'))
    if j%1000 == 0:
        print(j)


lookup1 = coupon_train.columns[4:17]
lookup2 = coupon_train.columns[24:73]
new_features = ['same_GENRE_count' , 'same_GENRE_DISCOUNT_PRICE' ,'same_GENRE_PRICE_RATE',
                'same_ken_count', 'same_ken_DISCOUNT_PRICE' ,'same_ken_PRICE_RATE']

for j,i in enumerate(coupon_byweek[:,2]):
    values = np.repeat(np.nan,i.shape[0]*6).reshape((i.shape[0],6))
    new_data = pd.DataFrame(values, columns=new_features)
    for k in range(i.shape[0]):
        if not np.isnan(new_data.iloc[k]).any():
            continue
        else:
            a = i.iloc[k]
            index_1 = i[lookup1].eq(a[lookup1],axis = 1).all(1)
            index_2 = i[lookup2].eq(a[lookup2],axis = 1).all(1)
            same_GENRE_count = np.sum(index_1)
            same_GENRE_DISCOUNT_PRICE = np.mean(i[index_1]['DISCOUNT_PRICE'])
            same_GENRE_PRICE_RATE = np.mean(i[index_1]['PRICE_RATE'])
            same_ken_count = np.sum(index_2)
            same_ken_DISCOUNT_PRICE = np.mean(i[index_2]['DISCOUNT_PRICE'])
            same_ken_PRICE_RATE = np.mean(i[index_2]['PRICE_RATE'])
            new_data.iloc[k][new_features] = [same_GENRE_count,same_GENRE_DISCOUNT_PRICE,same_GENRE_PRICE_RATE,
                                              same_ken_count, same_ken_DISCOUNT_PRICE,same_ken_PRICE_RATE]
    coupon_byweek[j,2][new_features] = new_data.values
    print(j)



coupon_byweek_val = np.zeros((10,3),dtype= object)
for i in range(10):
    coupon_byweek_val[i] = [datetime(2012, 4, 19)+ i*timedelta(7),datetime(2012, 4, 19)+ (i+1)*timedelta(7),pd.DataFrame(columns = coupon_list_test.columns.delete(0))]
coupon_val['DISPFROM'] = pd.to_datetime(coupon_val['DISPFROM'])



for j,i in coupon_val.iterrows():
    row = np.argmax(map(lambda x: i['DISPFROM'] > x[0] and i['DISPFROM'] <= x[1], coupon_byweek_val[:,:2]))
    coupon_byweek_val[row][2] = coupon_byweek_val[row][2].append(i.drop('DISPFROM'))
    if j%1000 == 0:
        print(j)

for j,i in enumerate(coupon_byweek_val[:,2]):
    values = np.repeat(np.nan,i.shape[0]*6).reshape((i.shape[0],6))
    new_data = pd.DataFrame(values, columns=new_features)
    for k in range(i.shape[0]):
        if not np.isnan(new_data.iloc[k]).any():
            continue
        else:
            a = i.iloc[k]
            index_1 = i[lookup1].eq(a[lookup1],axis = 1).all(1)
            index_2 = i[lookup2].eq(a[lookup2],axis = 1).all(1)
            same_GENRE_count = np.sum(index_1)
            same_GENRE_DISCOUNT_PRICE = np.mean(i[index_1]['DISCOUNT_PRICE'])
            same_GENRE_PRICE_RATE = np.mean(i[index_1]['PRICE_RATE'])
            same_ken_count = np.sum(index_2)
            same_ken_DISCOUNT_PRICE = np.mean(i[index_2]['DISCOUNT_PRICE'])
            same_ken_PRICE_RATE = np.mean(i[index_2]['PRICE_RATE'])
            new_data.iloc[k][new_features] = [same_GENRE_count,same_GENRE_DISCOUNT_PRICE,same_GENRE_PRICE_RATE,
                                              same_ken_count, same_ken_DISCOUNT_PRICE,same_ken_PRICE_RATE]
    coupon_byweek_val[j,2][new_features] = new_data
    coupon_byweek_val[j,2][new_features] = new_data.values
    print(j)


coupon_byweek_test = np.zeros((1,3),dtype= object)
coupon_byweek_test[0,0] =  datetime(2012, 6, 24)
coupon_byweek_test[0,1] =  datetime(2012, 6, 30)
coupon_byweek_test[0,2] = pd.DataFrame(columns = coupon_list_test.columns.delete(0))
coupon_byweek_test[0,2] = coupon_byweek_test[0,2].append(coupon_list_test.drop('DISPFROM',1))


lookup1 = coupon_list_test.columns[4:17]
lookup2 = coupon_list_test.columns[26:73]

for j,i in enumerate(coupon_byweek_test[:,2]):
    values = np.repeat(np.nan,i.shape[0]*6).reshape((i.shape[0],6))
    new_data = pd.DataFrame(values, columns=new_features)
    for k in range(i.shape[0]):
        if not np.isnan(new_data.iloc[k]).any():
            continue
        else:
            a = i.iloc[k]
            index_1 = i[lookup1].eq(a[lookup1],axis = 1).all(1)
            index_2 = i[lookup2].eq(a[lookup2],axis = 1).all(1)
            same_GENRE_count = np.sum(index_1)
            same_GENRE_DISCOUNT_PRICE = np.mean(i[index_1]['DISCOUNT_PRICE'])
            same_GENRE_PRICE_RATE = np.mean(i[index_1]['PRICE_RATE'])
            same_ken_count = np.sum(index_2)
            same_ken_DISCOUNT_PRICE = np.mean(i[index_2]['DISCOUNT_PRICE'])
            same_ken_PRICE_RATE = np.mean(i[index_2]['PRICE_RATE'])
            new_data.iloc[k][new_features] = [same_GENRE_count,same_GENRE_DISCOUNT_PRICE,same_GENRE_PRICE_RATE,
                                              same_ken_count, same_ken_DISCOUNT_PRICE,same_ken_PRICE_RATE]
    coupon_byweek_test[j,2][new_features] = new_data
    coupon_byweek_test[j,2][new_features] = new_data.values
    print(j)

coupon_byweek_test[0,2].to_csv('coupon_test_append.csv',index = False)

np.save('coupon_test_append.npy',np.array(coupon_byweek_test[0,2]))
np.save('coupon_train_append.npy',coupon_byweek)
np.save('coupon_val_append.npy',coupon_byweek_val)

###

user = user_list['USER_ID_hash']
res = np.zeros((user.shape[0],2),dtype= object)

pred = pd.read_csv('../model/ona_last_shot.csv').values[:,1:]
for j,i in enumerate(user):
    index = (pred[:,1] == i)
    temp1 = pred[index,2]
    temp1_val = pred[index,0]
    temp2 = (np.argsort(pred[index,2])[-10:])[::-1]
    res[j] = [i,','.join(temp1_val[temp2]) ]
    if j%500 ==0:
        print(j,res[j])

for i,j in enumerate(res):
    res[i,1] = ' '.join(res[i,1].split(','))


output = pd.DataFrame(res,columns= ['USER_ID_hash','PURCHASED_COUPONS']).to_csv('model_12.csv',index = False)
with open('output11.csv','wb') as f:
    csv.writer(f).writerows()

out= []
pred = np.load('test_sample_2.npy').item()
for i in user:
    if i in pred.keys():
        out.append([i,' '.join(pred[i])])
    else:
        out.append([i,''])

pd.DataFrame(out,columns= ['USER_ID_hash','PURCHASED_COUPONS']).to_csv('model_12.csv',index = False)

###