# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics 
import gc
import glob
#%%
#hdd = pd.read_csv('D:\TraceData\harddrive.csv')
#hdd = hdd.loc[:, ~hdd.isnull().all()]
hdd_all = pd.read_csv('D:\\TraceData\\try\\2017-01-01.csv')
hdd_all = hdd_all.query('model == "ST4000DM000"') 
hdd_all = hdd_all.loc[:, ~hdd_all.isnull().all()]
f = glob.iglob(r'D:\TraceData\harddisktest\data_Q1_2017\*.csv')
for disk in f:
    print(disk)
    hdd = pd.read_csv(disk)
    hdd = hdd.query('model == "ST4000DM000"')
    hdd = hdd.loc[:, ~hdd.isnull().all()]
    if hdd.shape[1] != hdd_all.shape[1]:
        print("ERROR!!")
    hdd_all = hdd_all.append(hdd, ignore_index=True)
#%%

#%%

