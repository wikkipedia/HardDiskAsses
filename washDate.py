# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:41:35 2017

@author: WIKKI
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import gc
#%%
gc.collect()
#%%
Q1 = hdd_all
#%%针对2017年一季度计算使用
Q1.loc[:,'date'] = pd.to_datetime(Q1.loc[:,'date'])
Q1['date'] = Q1['date'].dt.dayofyear
#%%
Q2 = hdd_all
#%%针对2017年一季度计算使用
Q2.loc[:,'date'] = pd.to_datetime(Q2.loc[:,'date'])
Q2['date'] = Q2['date'].dt.dayofyear + 366
first_day = 367
#%%下一季度的第一天
first_day = Q2.loc[0,'date']
first_day = pd.to_datetime(first_day)
first_day = first_day.dayofyear
#%%将两个季度的日志合成为一个
frames = [Q1,Q2]
Trace_Log = pd.concat(frames, ignore_index=True)
temp = Trace_Log[Trace_Log['failure']==1]
temp.isnull().any()
#%%绘制硬盘发生故障时间的散点图
temp.loc[:,'date'] = pd.to_datetime(temp.loc[:,'date'])
temp['day_of_year'] = temp['date'].dt.dayofyear
temp.plot(kind='scatter', x='day_of_year', y='failure', title='Hard drive failures over time')
plt.show()
#%%筛选出发生在下一季度的故障
temp.loc[:,'date'] = pd.to_datetime(temp.loc[:,'date'])
temp['date'] = temp['date'].dt.dayofyear
temp = temp[temp['date']>=first_day]
#%%找到故障硬盘的序列号，并筛选出故障硬盘的日志条目
failed_serial_num = temp.loc[:,'serial_number']
failed_disk = Trace_Log[Trace_Log['serial_number'].isin(failed_serial_num)]
#%%将日期转化为天数
failed_disk.loc[:,'date'] = pd.to_datetime(failed_disk.loc[:,'date'])
failed_disk['date'] = failed_disk['date'].dt.dayofyear
failed_disk_sort = failed_disk.sort_values(by=['serial_number','date'],ascending=[True,False])
failed_disk_sort['left_day'] = 0
#%%计算距离发生故障的剩余天数，并保存在left_day列中;同时记录硬盘故障后产生的日志记录
pre = 0
pre_serial = 'begin'
error = []
idx = []
for iter in failed_disk_sort.index:
    serial = failed_disk_sort.loc[iter, 'serial_number']
    if failed_disk_sort.loc[iter, 'failure'] == 1:
        pre = failed_disk_sort.loc[iter, 'date']
        pre_serial = serial
        failed_disk_sort.loc[iter, 'left_day'] = 0
    else:
        if serial == pre_serial:
            failed_disk_sort.loc[iter, 'left_day'] =  pre - failed_disk_sort.loc[iter, 'date']
        else:
            failed_disk_sort.loc[iter, 'left_day'] = -1
            error.append(serial)
            idx.append(iter)
#%%删除故障后忍产生的记录
failed_disk_sort.drop(idx, axis=0, inplace=True)
failed_disk_sort = failed_disk_sort.reset_index(drop=True)
#%%
temp = failed_disk_sort.groupby('serial_number')
temp_max_left = temp['left_day'].max()
temp_min_left = temp['left_day'].min()
#%%筛选出剩余天数小于40天的日志条目
failed_disk = failed_disk_sort[failed_disk_sort['left_day']<=50]
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
#temp = Y17
temp['status'] = Series(map(lambda x: healthmap[x], temp['left_day']))
#%%找到值始终不变的SMART属性
temp = Y17.dropna()
col = []
for iter in temp.columns:
    if temp[iter].value_counts().shape[0] == 1:
        col.append(iter)
temp.drop(col,axis=1,inplace=True)