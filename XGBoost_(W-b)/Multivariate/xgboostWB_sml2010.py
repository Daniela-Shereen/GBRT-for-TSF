# -*- coding: utf-8 -*-
"""DARNN_SML2010.ipynb
"""

import sys
sys.version
#Import Libraries
import pandas as pd
import os
#import matplotlib
#import matplotlib.pyplot as plt
import random
# %matplotlib inline
import shutil
import itertools
import numpy as np
import re
from random import shuffle

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

num_periods_output = 1 #to predict
num_periods_input=10 #input



ALL_Test_Data=[]
ALL_Test_Prediction=[]

"""## preprocessing"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preprocessing(df_,num_features):
    #,'Month','Day','Hour','Minute'
    # select feature
    df_['Date']=pd.to_datetime(df_['Date']) #, format='%d/%m/%Y'
    df_['day']=df_['Date'].dt.day
    df_['month']=df_['Date'].dt.month
    df_['year']=df_['Date'].dt.year
    df_['Time']=pd.to_datetime(df_['Time']) #, format='%h:%m'
    df_['hour']=df_['Time'].dt.hour
    df_['min']=df_['Time'].dt.minute
    cols=df_.columns
    
    Train=df_.iloc[0:3600,:]
    Test=df_.iloc[3600-num_periods_input:,:]
    Train=Train.fillna(Train.mean())
    Test=Test.fillna(Test.mean())
    ################################################encoding########################
    Train=Train[['Temperature_Habitacion_Sensor','day','month','year','hour','min','Temperature_Comedor_Sensor', 
                 'Weather_Temperature','CO2_Comedor_Sensor' ,'CO2_Habitacion_Sensor', 'Humedad_Comedor_Sensor', 
                 'Humedad_Habitacion_Sensor','Lighting_Comedor_Sensor','Lighting_Habitacion_Sensor','Precipitacion', 
                 'Meteo_Exterior_Crepusculo' ,'Meteo_Exterior_Viento','Meteo_Exterior_Sol_Oest','Meteo_Exterior_Sol_Est', 
                 'Meteo_Exterior_Sol_Sud', 'Meteo_Exterior_Piranometro','Exterior_Entalpic_1', 
                 'Exterior_Entalpic_2','Exterior_Entalpic_turbo','Temperature_Exterior_Sensor','Humedad_Exterior_Sensor','Day_Of_Week']]
    
    Train=Train.values
    Train = Train.astype('float32')
    #################################################################################
    
    Test=Test[['Temperature_Habitacion_Sensor','day','month','year','hour','min','Temperature_Comedor_Sensor', 
                 'Weather_Temperature','CO2_Comedor_Sensor' ,'CO2_Habitacion_Sensor', 'Humedad_Comedor_Sensor', 
                 'Humedad_Habitacion_Sensor','Lighting_Comedor_Sensor','Lighting_Habitacion_Sensor','Precipitacion', 
                 'Meteo_Exterior_Crepusculo' ,'Meteo_Exterior_Viento','Meteo_Exterior_Sol_Oest','Meteo_Exterior_Sol_Est', 
                 'Meteo_Exterior_Sol_Sud', 'Meteo_Exterior_Piranometro','Exterior_Entalpic_1', 
                 'Exterior_Entalpic_2','Exterior_Entalpic_turbo','Temperature_Exterior_Sensor','Humedad_Exterior_Sensor','Day_Of_Week']]
    Test=Test.values
    Test = Test.astype('float32')
    #################################################################################
    Number_Of_Features=num_features
    split=num_periods_output+num_periods_input
    
    ####################### CUT THE PORTION of the data that we are working on 
        
    #############################  Normalization on train  #############
    
    #print('Len o training   ',Train)
    Train = Train.astype('float32')
    normalizer = StandardScaler().fit(Train)
    Train=normalizer.transform(Train)

    ############################################ TRAIN minibatches ##################################
    
    end=len(Train)
    start=0
    next=0
    x_batches=[]
    y_batches=[]
    
    count=0
    print('lennnn',len(Train))
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
    
    ###########################################TEST Normalization################################

    Test = Test.astype('float32')
    Test=normalizer.transform(Test) 

    ############################################ TEST minibatches ##################################
    end_test=len(Test)
    start_test=0
    next_test=0
    x_testbatches=[]
    y_testbatches=[]
    
    #print('lennnn',len(Train))
    while start_test+(limit)<=end_test:
        next_test=start_test+num_periods_input
        x_testbatches.append(Test[start_test:next_test,:])
        y_testbatches.append(Test[next_test:next_test+num_periods_output,0])
        start_test=start_test+num_periods_output
    x_testbatches=np.asarray(x_testbatches)
    #print('x test-------------',len(x_testbatches))

    x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features)
    y_testbatches=np.asarray(y_testbatches)
    #print('yyyy=======',[len(s) for s in y_testbatches])
    y_testbatches = y_testbatches.reshape(len(y_testbatches), num_periods_output) 
    y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1) 
    print('len Test',len(Test))
    print('len xTestbatches',len(x_testbatches))
    
    return x_batches, y_batches, x_testbatches, y_testbatches

data_path1=r'/GBRT-for-TSF/Data/Multivariate/SML_Data/SML1.txt'
data_path2=r'/GBRT-for-TSF/Data/Multivariate/SML_Data/SML2.txt'

data_All=pd.DataFrame()
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]

range_list = [1]
data1=pd.read_csv(data_path1,sep=' ')
data2=pd.read_csv(data_path2,sep=' ')
#print('========================================')
#print(data2.head)
header=list(data2.columns.values)
header=[s.split(':')[1] for s in header]
#print(header)
data = data1.append(data2,ignore_index=True)
data.columns=header

x_batches_Full, y_batches_Full,X_Test_Full,Y_Test_Full=preprocessing(data,27)
#---------------------shuffle minibatches X and Y together-------------------------------------
combined = list(zip(x_batches_Full, y_batches_Full))
random.shuffle(combined)
shuffled_batch_features, shuffled_batch_y = zip(*combined)


#xgboost part
print(len(x_batches_Full))
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
    

print(len(All_Training_Instances[0]))

#=================Testing=====================
All_Testing_Instances=[]

#=============== change each window into Instance =================================

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
print(len(All_Testing_Instances[0]))
#===========================calling MultiOutput XGoost=========================
All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances),len(All_Testing_Instances[0])))
Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full),num_periods_output))

#========== reshape train ==============================
All_Training_Instances=np.reshape(All_Training_Instances, (len(All_Training_Instances),len(All_Training_Instances[0])))
shuffled_batch_y=np.reshape(shuffled_batch_y, (len(shuffled_batch_y),num_periods_output))



print(All_Training_Instances.shape)
model=xgb.XGBRegressor(learning_rate =0.09,
 n_estimators=150,
 max_depth=3,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.95,
 colsample_bytree=0.95,
 scale_pos_weight=0.8,
 seed=42,silent=False)

multioutput=MultiOutputRegressor(model).fit(All_Training_Instances,shuffled_batch_y)


print('Fitting Done!')

prediction=multioutput.predict(All_Testing_Instances)
print('prediction ',prediction.shape)
print('test ',Y_Test_Full.shape)
# =========================== MAPE ===============================
def mean_absolute_percentage_error(y_true, y_pred): 
    a=(y_true - y_pred)
    b=y_true
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return np.mean(np.abs(c)) 
# =================================================================
MSE=np.mean(( prediction- Y_Test_Full)**2)
print('RMSE: ',MSE**0.5)
MAE=np.mean(np.abs( prediction- Y_Test_Full))   
print('MAE: ',MAE) 
MAPE=mean_absolute_percentage_error(Y_Test_Full,prediction)
print('MAPE: ',MAPE)
