# -*- coding: utf-8 -*-
"""XGBoostWB_Forecasting_Using_Hybrid_DL_Urban_Dataset.ipynb
"""
import sys
sys.version
#Import Libraries
import itertools
import pandas as pd
import numpy as np
import os
#import matplotlib
#import matplotlib.pyplot as plt
import random
# %matplotlib inline
import shutil

from random import shuffle

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
#TF Version
tf.__version__

num_periods_output = 6 #to predict
num_periods_input=1 #input



ALL_Test_Data=[]
ALL_Test_Prediction=[]

"""## preprocessing"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

def preprocessing(df_,num_features):
  
    cols=df_.columns
    Train=df_[(df_.year==2014)]
    Test=df_[(df_.year==2015)]
    Train=Train.fillna(Train.mean())
    Test=Test.fillna(Test.mean())
    ################################################encoding########################
    Train=Train[['PM25_Concentration','station_id','year',	'month','day',	'hour', 'PM10_Concentration','NO2_Concentration', 'CO_Concentration', 'O3_Concentration',
       'SO2_Concentration',  'temperature','pressure', 'humidity', 'wind_speed', 'wind_direction']]
    
    Train=Train.values
    Train = Train.astype('float32')
    #################################################################################
    
    Test=Test[['PM25_Concentration','station_id','year',	'month','day',	'hour', 'PM10_Concentration','NO2_Concentration', 'CO_Concentration', 'O3_Concentration',
       'SO2_Concentration',  'temperature','pressure', 'humidity', 'wind_speed', 'wind_direction']]
    Test=Test.values
    Test = Test.astype('float32')
    #################################################################################
    Number_Of_Features=num_features
    split=num_periods_output+num_periods_input
        
    #############################  Normalization on training #############
    PM_Train=Train[:,0]
    Train=np.delete(Train,[0],1)
    #normalizing data
    print('Len of training   ',Train)
    Train = Train.astype('float32')
    normalizer = MinMaxScaler().fit(Train)
    #normalizer = Normalizer().fit(Train)
    Train=normalizer.transform(Train)

    PM_Train=np.reshape(PM_Train,(len(PM_Train),1))
    Train=np.append(PM_Train,Train, axis=1)
    ############################################ TRAIN windows ##################################
    
    end=len(Train)
    start=0
    next=0
    x_batches=[]
    y_batches=[]
    
    count=0
    #print('len',len(Train))
    limit=num_periods_output+num_periods_input
    while start+(limit)<=end:
        next=start+num_periods_input
        x_batches.append(Train[start:next,:])
        y_batches.append(Train[next:next+num_periods_output,0])
        start=start+1
    x_batches=np.asarray(x_batches)
    #print('x-------------',len(x_batches))
    x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)   
    y_batches=np.asarray(y_batches)
    #print('y=======',len(y_batches))
    y_batches = y_batches.reshape(-1, num_periods_output, 1)   
    #print('len x_batches ',len(x_batches))
    
    ###########################################TEST normalization#####################################
    PM_Test=Test[:,0]
    Test=np.delete(Test,[0],1)

    Test = Test.astype('float32')
    Test=normalizer.transform(Test) 

    PM_Test=np.reshape(PM_Test,(len(PM_Test),1))
    Test=np.append(PM_Test,Test, axis=1)
    #------------------
    ############################################ TEST windows ##################################
    end_test=len(Test)
    start_test=0
    next_test=0
    x_testbatches=[]
    y_testbatches=[]
    
    #print('len',len(Train))
    while start_test+(limit)<=end_test:
        next_test=start_test+num_periods_input
        x_testbatches.append(Test[start_test:next_test,:])
        y_testbatches.append(Test[next_test:next_test+num_periods_output,0])
        start_test=start_test+num_periods_output
    x_testbatches=np.asarray(x_testbatches)
    #print('x-------------',len(x_testbatches))

    x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features)
    y_testbatches=np.asarray(y_testbatches)
    #print('yyyy=======',len(y_testbatches))
    y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1) 
    print('len Test',len(Test))
    print('len xTestbatches',len(x_testbatches))
    
    return x_batches, y_batches, x_testbatches, y_testbatches

data_path1=r'/GBRT-for-TSF/Data/Multivariate/UrbanAirQualityData/meteorology.csv'
data_path2=r'/GBRT-for-TSF/Data/Multivariate/UrbanAirQualityData/airquality.csv'

data_All=pd.DataFrame()
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]

range_list = [1]
data1=pd.read_csv(data_path1)
data2=pd.read_csv(data_path2)
#print(list(data1.columns.values))
#print('========================================')
#print(data2.head)
header=list(data2.columns.values)
#print(header) 
data2['time'] = pd.to_datetime(data2['time'])
data=pd.DataFrame(data2,columns=header)
data['year']=data2['time'].dt.year
data['month']=data2['time'].dt.month
data['day']=data2['time'].dt.day
data['hour']=data2['time'].dt.hour
data['temperature']=data1['temperature']
data['pressure']=data1['pressure']
data['humidity']=data1['humidity']
data['wind_speed']=data1['wind_speed']
data['wind_direction']=data1['wind_direction']
#print(data.columns)

x_batches_Full, y_batches_Full,X_Test_Full,Y_Test_Full=preprocessing(data,16)
#---------------------shuffle windows X and Y together-------------------------------------
#print(len(x_batches_Full),'     length of all file : ',len(y_batches_Full))
combined = list(zip(x_batches_Full, y_batches_Full))
random.shuffle(combined)
shuffled_batch_features, shuffled_batch_y = zip(*combined)



#xgboost part
#print(len(x_batches_Full))
All_Training_Instances=[]
 
#=============== change each window into Instance =================================
for i in range(0,len(shuffled_batch_features)):
    hold=[]
    temp=[]
    for j in range(0,len(shuffled_batch_features[i])):
      #print(len(hold))
      
      if j==(len(shuffled_batch_features[i])-1):
          hold=np.concatenate((hold, shuffled_batch_features[i][j][:]), axis=None)
          
      else:
          hold=np.concatenate((hold, shuffled_batch_features[i][j][0]), axis=None)
          
    #print(len(hold))
    All_Training_Instances.append(hold)
    

#print(len(All_Training_Instances[0]))


#=============== change each testing window into Instance =================================
All_Testing_Instances=[]
#print(len(X_Test_Full))
for i in range(0,len(X_Test_Full)):
  hold=[]
  temp=[]
  for j in range(0,len(X_Test_Full[i])):
       #print(len(hold))
      if j==(len(X_Test_Full[i])-1):
          hold=np.concatenate((hold, X_Test_Full[i][j][:]), axis=None)
      else:
          hold=np.concatenate((hold, X_Test_Full[i][j][0]), axis=None)
   
  All_Testing_Instances.append(hold)

#prediction=multioutput.predict(All_Testing_Instances)
#print(len(All_Testing_Instances[0]))
#===========================calling MultiOutput XGoost=========================
All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances),len(All_Testing_Instances[0])))
Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full),num_periods_output))

#========== reshape train ==============================
All_Training_Instances=np.reshape(All_Training_Instances, (len(All_Training_Instances),len(All_Training_Instances[0])))
shuffled_batch_y=np.reshape(shuffled_batch_y, (len(shuffled_batch_y),num_periods_output))



#print(All_Training_Instances.shape)
model=xgb.XGBRegressor(learning_rate =0.025,
 n_estimators=250,
 max_depth=2,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.98,
 colsample_bytree=0.98,
 scale_pos_weight=0.8,
 seed=42,silent=False)

multioutput=MultiOutputRegressor(model).fit(All_Training_Instances,shuffled_batch_y)


print('Fitting Done!')

prediction=multioutput.predict(All_Testing_Instances)
print('prediction ',prediction.shape)
print('test ',Y_Test_Full.shape)
MSE=np.mean(( prediction- Y_Test_Full)**2)
print('RMSE: ',MSE**0.5)
MAE=np.mean(np.abs( prediction- Y_Test_Full)) 
#print('With Features for {} weeks'.format(No_Of_weeks))   
print('MAE: ',MAE)
