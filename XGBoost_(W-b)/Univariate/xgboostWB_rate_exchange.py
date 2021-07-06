# -*- coding: utf-8 -*-
"""XGBoostWB-Rate_Exchange
"""
import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np
import os
#import matplotlib
#import matplotlib.pyplot as plt
import random
# %matplotlib inline
import shutil
from random import shuffle
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn import preprocessing

num_periods_output = 24 #to predict
num_periods_input=24 #input

ALL_Test_Data=[]
ALL_Test_Prediction=[]

"""## preprocessing"""

def New_preprocessing(TimeSeries):
   #print(len(TimeSeries))
   Data=[]
   start_date=datetime(1990, 1, 1,00,00,00) # define start date
   for i in range(0,len(TimeSeries)):
      record=[]
      record.append(TimeSeries[i]) #adding the xchangerate value
      record.append(start_date.month)
      record.append(start_date.day)
      #record.append(start_date.hour)
      #record.append(start_date.minute)
      record.append(start_date.weekday())
      record.append(start_date.timetuple().tm_yday)
      record.append(start_date.isocalendar()[1])
      #print(start_date.month,' ',start_date.day,' ',start_date.hour,' ',start_date.weekday(),' ',start_date.timetuple().tm_yday,' ',start_date.isocalendar()[1])
      start_date=start_date+ timedelta(days=1)
      #print('year',start_date.year,'Month:',start_date.month,' day:',start_date.day,' hour:',start_date.hour)
      Data.append(record)
   ########## change list of lists to df ################
   headers=['pems','month','day','day_of_week','day_of_year','week_of_year']
   Data_df = pd.DataFrame(Data, columns=headers)
   #print(Data_df)
   sub=Data_df.iloc[:,1:]
   New_sub= preprocessing.minmax_scale(sub, feature_range=(-0.5, 0.5))
   #Normalize features to be from -0.5 to 0.5 as mentioned in the paper
   Normalized_Data_df=pd.DataFrame(pd.np.column_stack([Data_df.iloc[:,0],New_sub]), columns=headers)
   #print(Normalized_Data_df)
   #################################################################################################
   # cut training and testing 
   train_split=np.floor(len(Normalized_Data_df)*0.8)   # 60 % training
   #train_split=180
   train_split=int(train_split-(train_split%(num_periods_output+num_periods_input)))
   #print('-------------------',train_split)
   Train=Normalized_Data_df.iloc[0:train_split,:]
   Train=Train.values
   Train = Train.astype('float32')
   print('Traing length :',len(Train))
   total=len(Normalized_Data_df)
   test_split=np.floor(len(Normalized_Data_df)*0.2)   # 20 % testing
   #test_split=20
   #print('-------------------test: ',test_split)
   test_split=int(test_split-(test_split%(num_periods_output+num_periods_input)))
   Test=Normalized_Data_df.iloc[(total-test_split-num_periods_input):,:]
   Test=Test.values
   Test = Test.astype('float32')
   print('Traing length :',len(Test))
   Number_Of_Features=6
   ############################################ Windowing ##################################
   end=len(Train)
   start=0
   next=0
   x_batches=[]
   y_batches=[]  
   count=0
   #print('lennnn',len(Train))
   limit=max(num_periods_input,num_periods_output)
   while next+(limit)<end:
        next=start+num_periods_input
        x_batches.append(Train[start:next,:])
        y_batches.append(Train[next:next+num_periods_output,0])
        start=start+1
   y_batches=np.asarray(y_batches)
   y_batches = y_batches.reshape(-1, num_periods_output, 1) 
   #print('Length of y batches :',len(y_batches),' ',num_periods_input,' ',num_periods_output)
   #print(x_batches)
   x_batches=np.asarray(x_batches) 
   x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)   
   #print('len x_batches ',len(x_batches))
   ############################################ Windowing ##################################
   end_test=len(Test)
   start_test=0
   next_test=0
   x_testbatches=[]
   y_testbatches=[]
   while next_test+(limit)<end_test:
        next_test=start_test+num_periods_input
        x_testbatches.append(Test[start_test:next_test,:])
        y_testbatches.append(Test[next_test:next_test+num_periods_output,0])
        start_test=start_test+num_periods_input
   y_testbatches=np.asarray(y_testbatches)
   y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1)   
   x_testbatches=np.asarray(x_testbatches)
   x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features) 
   #print(' xTestbatches',len(x_testbatches),' yTestbatches',len(y_testbatches))
   return x_batches, y_batches, x_testbatches, y_testbatches

data=pd.read_csv('/GBRT-for-TSF/Data/Univariate/exchange_rate.txt',sep=',',header=None)

data=pd.DataFrame(data)
data=data.T
#print(data.shape)
#print('Number of timeseries: ',len(data))
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]
for i in range(0,len(data)):
    print('Time series: ',i)
    x_batches=[]
    y_batches=[]
    X_Test=[]
    Y_Test=[]
    TimeSeries=data.iloc[i,:]
    #New_preprocessing(TimeSeries)
    #TimeSeries=np.arange(1,7588,1)
    #print(TimeSeries[7586:])
    x_batches, y_batches,X_Test,Y_Test=New_preprocessing(TimeSeries)          
    for element1 in (x_batches):
        x_batches_Full.append(element1)
            
    for element2 in (y_batches):
        y_batches_Full.append(element2)
                        
    for element5 in (X_Test):
        X_Test_Full.append(element5)
            
    for element6 in (Y_Test):
        Y_Test_Full.append(element6)
    #---------------------shuffle windows  X and target Y together-------------------------------------
#print(len(x_batches_Full),'     length of all file : ',len(y_batches_Full))
combined = list(zip(x_batches_Full, y_batches_Full))
random.shuffle(combined)
shuffled_batch_features, shuffled_batch_y = zip(*combined)

#xgboost part
print(len(x_batches_Full))
All_Training_Instances=[]
 
#=============== flatten each training window into Instance =================================
for i in range(0,len(shuffled_batch_features)):
    hold=[]
    temp=[]
    for j in range(0,len(shuffled_batch_features[i])):
      #**************** to run without features --> comment if else condition **************************
      #print(len(hold))
      if j==(len(shuffled_batch_features[i])-1):
        hold=np.concatenate((hold, shuffled_batch_features[i][j][:]), axis=None)   
      else:
        hold=np.concatenate((hold, shuffled_batch_features[i][j][0]), axis=None)
        
    All_Training_Instances.append(hold)
    

print(len(All_Training_Instances[0]))


#=============== flatten each testing window into Instance =================================
All_Testing_Instances=[]
print(len(X_Test_Full))
for i in range(0,len(X_Test_Full)):
  hold=[]
  temp=[]
  for j in range(0,len(X_Test_Full[i])):
       #****************  to run without features --> comment if else condition **************************
       #print(len(hold))
      if j==(len(X_Test_Full[i])-1):
          hold=np.concatenate((hold, X_Test_Full[i][j][:]), axis=None)
      else:
          hold=np.concatenate((hold, X_Test_Full[i][j][0]), axis=None)
   
  All_Testing_Instances.append(hold)

print(len(All_Testing_Instances[0]))

#=========================== final shape check =========================
All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances),len(All_Testing_Instances[0])))
Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full),num_periods_output))

All_Training_Instances=np.reshape(All_Training_Instances, (len(All_Training_Instances),len(All_Training_Instances[0])))
shuffled_batch_y=np.reshape(shuffled_batch_y, (len(shuffled_batch_y),num_periods_output))



#=========================== CALLING XGBOOST ===========================
model=xgb.XGBRegressor(learning_rate =0.07,
 n_estimators=80,
 max_depth=3,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.97,
 colsample_bytree=0.97,
 scale_pos_weight=1,
 seed=42,silent=False)

multioutput=MultiOutputRegressor(model).fit(All_Training_Instances,shuffled_batch_y)


print('Fitting Done!')

#============================== PREDICTION ===============================
prediction=multioutput.predict(All_Testing_Instances)

MAPE=np.mean((np.abs(prediction- Y_Test_Full)/np.abs(Y_Test_Full)))
WAPE=np.sum(np.abs(prediction- Y_Test_Full))/np.sum(np.abs(Y_Test_Full))
#print('With Features for {} weeks'.format(No_Of_weeks)) 
#print('MAPE: ',MAPE)
#print('WAPE: ',WAPE)

MSE=np.mean((prediction- Y_Test_Full)**2) 
MAE=np.mean(np.abs((prediction- Y_Test_Full)))
MAPE=np.mean((np.abs(prediction- Y_Test_Full)/np.abs(Y_Test_Full))) 
WAPE=np.sum(np.abs(prediction- Y_Test_Full))/np.sum(np.abs(Y_Test_Full))
print('#~MAE: ',MAE) 
#print('With Features for {} weeks'.format(No_Of_weeks))    
print('MAPE: ',MAPE)   
print('WAPE: ',WAPE)      
 
#print('With Features for {} weeks'.format(No_Of_weeks)) 
print('RMSE: ',MSE**0.5)



