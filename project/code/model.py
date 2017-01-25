import pandas as pd
import numpy as np

coupon_list_train = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv')
coupon_list_test = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv')
user_list = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/user_list.csv')
coupon_purchases_train = pd.read_csv("/Users/cryan/Google Drive/1003_project/data/coupon_detail_train.csv")



temp_train = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_train.npy')[:,-2:]
temp_test = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_test.npy')[:,-2:]
coupon_list_train['lati'] = pd.DataFrame(temp_train[:,0])
coupon_list_train['long'] = pd.DataFrame(temp_train[:,1])
coupon_list_test['lati'] = pd.DataFrame(temp_test[:,0])
coupon_list_test['long'] = pd.DataFrame(temp_test[:,1])


train_idx = np.load('train_idx.npy')
val_idx = np.load('test_idx.npy')

coupon_list_val  = coupon_list_train[val_idx]
coupon_list_train = coupon_list_train[train_idx]

#coupon_list_test = coupon_list_val





### merge to obtain (USER_ID) <-> (COUPON_ID with features) training set
purchased_coupons_train = coupon_purchases_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='right')
# features = ['COUPON_ID_hash', 'USER_ID_hash',
#             'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
#             'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
#             'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
#             'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']

features = ['COUPON_ID_hash', 'USER_ID_hash',
            'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'lati','long']


purchased_coupons_train = purchased_coupons_train[features]

coupon_list_val['USER_ID_hash'] = 'dummyuser'
coupon_list_val = coupon_list_val[features]
combined = pd.concat([purchased_coupons_train, coupon_list_val], axis=0)
combined['DISCOUNT_PRICE'] = 1 / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100) ** 2
features.extend(['DISCOUNT_PRICE', 'PRICE_RATE'])

categoricals = ['GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,
                                    dummy_na=False)

continuous = list(set(features) - set(categoricals))
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)
combined = combined.fillna(1)

train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1)

train_dropped_coupons = train.drop('COUPON_ID_hash', axis=1)
user_profiles = train_dropped_coupons.groupby(by='USER_ID_hash').apply(np.mean)

user_profiles_full = np.zeros((22873, 26))
for i,j in enumerate(user_list['USER_ID_hash'].values):
    if j in user_profiles.index:
        user_profiles_full[i] = user_profiles.loc[j]




FEATURE_WEIGHTS = {
    'GENRE_NAME': 1,
    'DISCOUNT_PRICE': 1,
    'PRICE_RATE': 1,
    'USABLE_DATE_': 1,
    'lati':1,
    'long':2
}

#construct error

val_hist = []
for j,i in enumerate(test['COUPON_ID_hash'].values):
    temp = coupon_purchases_train['COUPON_ID_hash'].values == i
    if np.sum(temp) == 0:
        continue
    else:
        val_hist.append([j,coupon_purchases_train[temp].values[:,-2]])



def simi_error(w,user_profiles_full):
    error =0.0
    num = 0
    for i in val_hist:
        coupon_cur = test.values[i[0]][1:]
        user_index = map(lambda x:user_hash[x], i[1])
        for ele  in user_index:
            if np.sum(user_profiles_full[ele]) != 0:
                error+=np.dot(user_profiles_full[ele]*w,coupon_cur )/(np.linalg.norm(user_profiles_full[ele]*w)*np.linalg.norm(coupon_cur))
                num+=1
    return error/num

def run_train(epoch = 10):
    w = np.ones(26)/np.sum(np.ones(26))
    curr_err = 0
    n = 0
    w_cur = w.copy()

    while n<=epoch:
        for index in np.random.permutation(26):
            for trial in np.linspace(0,1*10**(-n),11) + w_cur[index]:
                w_cur = w.copy()
                w_cur[index] = trial
                w_cur = w_cur/np.sum(w_cur)
                error = simi_error(w_cur,user_profiles_full)
                if error > curr_err:
                    curr_err = error
                    w = w_cur
            print(n,index,curr_err)
        n+=1
    return w

w =  run_train(2)

############
coupon_list_train = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/coupon_list_train.csv')
coupon_purchases_train = pd.read_csv("/Users/cryan/Google Drive/1003_project/data/coupon_detail_train.csv")
coupon_list_test = pd.read_csv('/Users/cryan/Google Drive/1003_project/data/test/coupon_list_test.csv')




temp_train = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_train.npy')[:,-2:]
temp_test = np.load('/Users/cryan/Google Drive/1003_project/code/coupon_matrix_test.npy')[:,-2:]
coupon_list_train['lati'] = pd.DataFrame(temp_train[:,0])
coupon_list_train['long'] = pd.DataFrame(temp_train[:,1])
coupon_list_test['lati'] = pd.DataFrame(temp_test[:,0])
coupon_list_test['long'] = pd.DataFrame(temp_test[:,1])


purchased_coupons_train = coupon_purchases_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='inner')
# features = ['COUPON_ID_hash', 'USER_ID_hash',
#             'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
#             'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
#             'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
#             'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']

features = ['COUPON_ID_hash', 'USER_ID_hash',
            'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'lati','long']


purchased_coupons_train = purchased_coupons_train[features]

coupon_list_test['USER_ID_hash'] = 'dummyuser'
coupon_list_test = coupon_list_test[features]
combined = pd.concat([purchased_coupons_train, coupon_list_test], axis=0)
combined['DISCOUNT_PRICE'] = 1 / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100) ** 2
features.extend(['DISCOUNT_PRICE', 'PRICE_RATE'])

categoricals = ['GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,
                                    dummy_na=False)

continuous = list(set(features) - set(categoricals))
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)
combined = combined.fillna(1)

train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1)

train_dropped_coupons = train.drop('COUPON_ID_hash', axis=1)
user_profiles = train_dropped_coupons.groupby(by='USER_ID_hash').apply(np.mean)






def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if col in colname:
            return weight
    raise ValueError

#W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
#            for colname in user_profiles.columns]
W = np.diag(w)

test_only_features = test.drop('COUPON_ID_hash', axis=1)
similarity_scores = np.dot(np.dot(user_profiles, W),
                           test_only_features.T)

coupons_ids = test['COUPON_ID_hash'].values
index = user_profiles.index
columns = [coupons_ids[i] for i in range(0, similarity_scores.shape[1])]
result_df = pd.DataFrame(index=index, columns=columns,
                      data=similarity_scores)

def get_top10_coupon_hashes_string(row):
    row.sort()
    return ' '.join(row.index[-10:][::-1].tolist())

output = result_df.apply(get_top10_coupon_hashes_string, axis=1)


output_df = pd.DataFrame(data={'USER_ID_hash': output.index,
                               'PURCHASED_COUPONS': output.values})
output_df_all_users = pd.merge(user_list, output_df, how='left', on='USER_ID_hash')
output_df_all_users.to_csv('cosine_sim_python.csv', header=True,
                           index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])






