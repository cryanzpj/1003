import numpy as np
import scipy as sp
import pandas as pd
from collections import defaultdict
from datetime import datetime
from geopy.geocoders import GoogleV3


def map_exactor(loc):
    if loc:
        geolocator = Nominatim()
        location = geolocator.geocode(loc)
        return [location.latitude,location.longitude]
    else:
        return ['','']


def prefecture(loc):
    res = defaultdict()
    for i in loc:
        if i[0] == "\xef\xbb\xbfPREF_NAME":
            continue
        else:
            res[i[0]] = [i[2],i[3].strip()]
    res[''] = ['','']
    return res




prefecture_dict = prefecture(np.array([i.split(',')
                     for i in open('/Users/cryan/Google Drive/1003_project/data/prefecture_locations.csv')]))

user = np.array([i.split(',') for i in open('/Users/cryan/Google Drive/1003_project/data/user_list.csv')],dtype = object)
user[:,-1] = [i.strip() for i in user[:,-1]]
user = np.hstack((user,np.zeros((user.shape[0],2))))
user[0,-2:] = ["latitude", "longitude"]
user[1:,6:] = np.array(map(lambda x:prefecture_dict[x],user[1:,4]))
user[1:,0] = np.array(map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S") , user[1:,0]))
user[1:,3] = np.array(map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S") if x != 'NA' else '' , user[1:,3]))



#featurelize trainining
data = pd.read_csv("/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv",sep=',',header = False)

for field in ['CAPSULE_TEXT', 'GENRE_NAME']:
    # Go through each possible value except the last one
    for value in data[field].unique()[0:-1]:
        # Create a new binary field
        data[field + "_" + value] = pd.Series(data[field] == value, dtype=int)

    # Drop the original field
    data = data.drop([field], axis=1)

data = data.drop(['DISPFROM','DISPEND','VALIDFROM','VALIDEND'],axis = 1)

#res = np.zeros((data.shape[0],2))



for i in xrange(17308,17308+2600):
    print(i)
    temp = data['small_area_name'][i].split('\xe3\x83\xbb')[0]
    loc = data['large_area_name'][i] +',' + data['ken_name'][i] +','+ temp
    geolocator = GoogleV3(api_key = 'AIzaSyARgiHVhYiT1E5zV7gqO6pqUgGwmTrjLs0',timeout = 10)
    location = geolocator.geocode(loc)
    res[i,:] = [location.latitude,location.longitude]

data['coupon_lati'] = res[:,0]
data['coupon_long'] = res[:,1]
data = data.drop(['large_area_name','ken_name','small_area_name'],axis =1)

data = data.fillna(0)

coupon_matrix_train = data.values
coupon_matrix_train = np.delete(coupon_matrix_train,14,1)

mean_train  = coupon_matrix_train.mean(axis = 0)
std_train = np.array(map(lambda x: np.sqrt(x),np.var(coupon_matrix_train,axis = 0)))

coupon_matrix_train = (coupon_matrix_train - mean_train)/std_train

#np.save('coupon_matrix_train.npy',coupon_matrix_train)

#featurlizing testing
data2_in =  pd.read_csv("/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv",sep=',',header = False)
data2_in = data2_in.drop(['DISPFROM','DISPEND','VALIDFROM','VALIDEND'],axis = 1)

data2 = pd.DataFrame(columns = data.columns)
for i in data2_in.iterrows():
    i = i[1]
    temp = i['small_area_name'].split('\xe3\x83\xbb')[0]
    loc = i['large_area_name'] +',' + i['ken_name'] +','+ temp
    geolocator = GoogleV3(api_key = 'AIzaSyARgiHVhYiT1E5zV7gqO6pqUgGwmTrjLs0',timeout = 10)
    location = geolocator.geocode(loc)
    temp = [location.latitude,location.longitude]

    if 'CAPSULE_TEXT' +'_'+ i[0] in data.columns[15:51]:
        i['CAPSULE_TEXT' +'_'+ i[0]] = 1
    if 'GENRE_NAME' + '_' + i[1] in data.columns[15:51]:
         i['GENRE_NAME' + '_' + i[1]] = 1

    i = i.drop(['CAPSULE_TEXT','GENRE_NAME','large_area_name','ken_name','small_area_name'])
    i['coupon_lati'] = temp[0]
    i['coupon_long'] = temp[1]
    data2 = data2.append(i)

data2 = data2.fillna(0)

coupon_matrix_test = data2.values
coupon_matrix_test =  np.delete(coupon_matrix_test,14,1)
coupon_matrix_test = (coupon_matrix_test - mean_train) / std_train
#np.save('coupon_matrix_test.npy',coupon_matrix_test)
c_1 = coupon_matrix_test[0,-1]/coupon_matrix_test[1,-1]
mu_long = (data2.values[1,-1]*c_1 - data2.values[0,-1])/(c_1-1)
std_long = (data2.values[0,-1] - mu_long)/coupon_matrix_test[0,-1]

c_2 = coupon_matrix_test[0,-2]/coupon_matrix_test[1,-2]
mu_lat = (data2.values[1,-2]*c_2 - data2.values[0,-1])/(c_2-1)
std_lat = (data2.values[0,-2] - mu_lat)/coupon_matrix_test[0,-2]

##train validation split

train = np.array(map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S") < t[-500], coupon_list_train['DISPFROM'].values))
test = np.array(map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S") >= t[-500], coupon_list_train['DISPFROM'].values))
coupon_list_train[train]
coupon_list_train[test]
np.save('train_idx.npy',train)
np.save('test_idx.npy',test)




