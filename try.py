# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:12:24 2017

@author: WIKKI
"""
import pandas as pd
import gc
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing

from sklearn.cross_validation import train_test_split

from keras.utils import np_utils

gc.collect()

#%%
from keras.utils import np_utils

#%%
Y16Q2 = hdd_all #通用为Q2
#%%
hdd = pd.read_csv('D:\TraceData\harddrive.csv')
hdd = hdd.query('model == "ST4000DM000"')
hdd = hdd.loc[:, ~hdd.isnull().all()]
#%%
frames = [Y16Q1, Y16Q2, Y16Q3, Y16Q4]
result = pd.concat(frames, ignore_index=True)
#%%
if all(Y16Q1.columns == Y16Q4.columns):
    print('true')
else:
    print('false')
#%%
temp = Y16Q2[Y16Q2['failure']==1]
temp.isnull().any()
    #%%
temp.loc[:,'date'] = pd.to_datetime(temp.loc[:,'date'])
temp['day_of_year'] = temp['date'].dt.dayofyear
#%%
temp.plot(kind='scatter', x='day_of_year', y='failure', title='Hard drive failures over time')
plt.show()
#%%
temp = Y16Q2.groupby('serial_number')
temp_last_day = temp.nth(-1)
#%%
uniq_serial_num = pd.Series(temp_last_day.index.unique())
uniq_serial_num.shape
#%%
test_ids = uniq_serial_num.sample(frac=0.25)
train = temp_last_day.query('index not in @test_ids')
test = temp_last_day.query('index in @test_ids')
#%%
train_labels = train['failure']
test_labels = test['failure']
train = train.drop('failure', axis=1)
test = test.drop('failure', axis=1)
#%%
failed = Y16Q2[Y16Q2['failure']==1]
failed_serial_num = failed.loc[:,'serial_number']
#serial_num = pd.Series(failed_serial_num.index.unique())
failed_disk = Y16Q2[Y16Q2['serial_number'].isin(failed_serial_num)]
#%%
failed_disk.loc[:,'date'] = pd.to_datetime(failed_disk.loc[:,'date'])
failed_disk['date'] = failed_disk['date'].dt.dayofyear
#%%
failed_disk_sort = failed_disk.sort_values(by=['serial_number','date'],ascending=[True,False])
failed_disk_sort['status'] = 0
#%%
temp = failed_disk_sort.groupby('serial_number')
temp_last_day = temp.nth(0)
serial_num = temp_last_day.index.tolist()
fail_date = tuple(temp_last_day['date'].tolist())
dict_date = dict(zip(serial_num, fail_date))
#%%
pre = 0
for iter in failed_disk_sort.index:
    if failed_disk_sort.loc[iter, 'failure'] == 1:
        pre = failed_disk_sort.loc[iter, 'date']
        failed_disk_sort.loc[iter, 'status'] = 0
    else:
        failed_disk_sort.loc[iter, 'status'] =  pre - failed_disk_sort.loc[iter, 'date']
#%%
temp = failed_disk_sort.groupby('serial_number')
temp_max_left = temp['status'].max()
#%%
tt = Q2[Q2['serial_number']=='Z305ANSH']
#%%
data = [[2000, 'Ohino', 1.5], [2001, 'Ohino', 1.7], [2002, 'Ohino', 3.6], [2001, 'Nevada', 2.4], [2002, 'Nevada', 2.9]]
data = np.array(data)
df = DataFrame(data, index=['one', 'two', 'three', 'four', 'five'], columns=['year', 'state', 'pop'])
print(df)
#%%
foo = df.values
foo = df.as_matrix()
foo = np.array(df)
print(foo)
foo = df.as_matrix(['year', 'pop'])
print(foo)
#%%
frames = [Q12,Q3,Q4]
Y16 = pd.concat(frames, ignore_index=True)
#%%生成剩余天数与健康状态的对应表
left = range(51)
health = [1]*3
health.extend([2]*3)
health.extend([3]*5)
health.extend([4]*6)
health.extend([5]*7)
health.extend([6]*8)
health.extend([7]*19)
healthmap = dict(zip(left,health))
#%%Y17的index不是从0开始的，要重排以后才能使用map函数
temp = Y17.reset_index(drop=True)
#%%根据剩余天数判定健康状态
temp['status'] = Series(map(lambda x: healthmap[x], temp['left_day']))
#%%
temp = Y16.groupby('serial_number')
temp_max_left = temp['left_day'].min()
#%%
temp = Q4[Q4['serial_number']=='Z300GZ0W']
#%%检查硬盘故障后是否还在产生日志记录
pre = 'begin'
error = []
for iter in Y17.index:
    serial = Y17.loc[iter, 'serial_number']
    if serial != pre:
        pre = serial 
        if Y17.loc[iter, 'failure'] == 0:
            error.append(serial)
#%%找到值始终不变的SMART属性
temp = Y16.dropna()
col = []
for iter in temp.columns:
    if temp[iter].value_counts().shape[0] == 1:
        col.append(iter)
temp.drop(col,axis=1,inplace=True)
#%%
tt = failed_disk_sort[failed_disk_sort['serial_number']=='Z300GZ0W']
#%%
tem = temp[temp['status'].isnull()]
#%%
#y_data = [c - 1 for c in y_data]
y_test = np_utils.to_categorical(y_data, 6)
#%%
Y16 = pd.read_pickle('/home/jwang/Documents/processed/Y2')
Y17 = pd.read_pickle('/home/jwang/Documents/processed/Y17')
#%%
frames = [Y16, Y17]
Y2 = pd.concat(frames, ignore_index=True)
Y2.to_pickle('/home/jwang/Documents/processed/Y2')
#%%注意Y16的index少掉一位，要重排
dr = ['smart_183_normalized','smart_183_raw','date','failure','left_day']
npdata = Y16.drop(dr,axis=1)
npdata = np.array(npdata)
#%%对各个feature的数值归一化
temp = npdata[:,1:-1]        
min_max_1 = preprocessing.MinMaxScaler()
temp = min_max_1.fit_transform(temp)
npdata[:,1:-1] = temp
#%%查找每个硬盘最后一条日志对应的行数
serial = []
idx = []
pre = 'begin'
l = len(npdata)
for i in range(0, l):
    if npdata[i,0] != pre:
        serial.append(pre)
        idx.append(i-1)
        pre = npdata[i,0]
idx.append(i)
serial.append(pre)
last_idx = dict(zip(serial,idx))
#%%制作timesteps为20的数据集
now = 'begin'
x_data = []
y_data = []
for i in range(0,l):
    if npdata[i,0] != now:
        now = npdata[i,0]
        last = last_idx[now]
    if npdata[i,-1]<7:
        end = min(i+19, last) + 1
        sliced = npdata[i:end,1:-1]
        x_data.append(sliced)
        y_data.append(npdata[i,-1])
#%% padding with zeros
data = x_data
for i in range(0,len(data)):
    result = data[i]
    result = np.pad(result, ((0,20-result.shape[0]),(0,0)), 'constant', constant_values=0)
    data[i] = result
x_data = np.array(data)
#%%划分测试集与训练集
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
#%%
temp = npdata[:,1]
temp.reshape(-1,1)
temp = preprocessing.minmax_scale(temp)
#%%
tt = data[45]
tt = tt[:,1:]
print(tt)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(tt)
#%%
Y16.to_pickle('/home/wikki/Documents/processed/Y16.pkl')
Y17.to_pickle('/home/wikki/Documents/processed/Y17.pkl')
#%%
x_data.save('x_data.npy',a)
#%%
Y16 = pd.read_pickle('/home/wikki/Documents/processed/Y16')
#%%
Y16.to_csv('Y16.csv')
Y17.to_csv('Y17.csv')
#%%
foo = np.array(df)
fo1 = np.array(df)
fo1[0,0] = 2017
ll = []
ll.append(foo)
ll.append(fo1)
#%%
dummy_y = np_utils.to_categorical(y_data, num_classes=6)
dummy_y = dummy_y[:,1:]
#%%
tt = np.array([[ 0.16436073,  0.1616614,   0.19381061,  0.15829226,  0.16597888,  0.15589602],[ 0.16255681,  0.16395631,  0.1929483,   0.16396037,  0.16037005,  0.15620814],[ 0.16461335,  0.16227967,  0.19619974,  0.15777504,  0.16453448,  0.15459777],[ 0.15914862,  0.16232671,  0.20141737,  0.16303335,  0.16228095,  0.15179311],[ 0.15729198,  0.16542646,  0.19662228,  0.16531138,  0.16239175,  0.15295611]])
t1 = tt[:,:5]
t2 = tt[:,5]
#%%
r = [[1,3,3]]
r.append([1,2,5])
r = np.asarray(r)
tt = np.mean(r, axis=0)
#%%
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
r = metrics.f1_score(y_true, y_pred, average=None).tolist()
r = r + metrics.recall_score(y_true, y_pred, average=None).tolist()

r.append(metrics.precision_score(y_true, y_pred, average='macro'))
#classification_report(y_true, y_pred, average='macro')  