# -*- coding: utf-8 -*-
"""DARNN_NASDAQ.ipynb
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
import itertools
import re
from random import shuffle

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
#TF Version



num_periods_output = 1 #to predict
num_periods_input=10 #input



ALL_Test_Data=[]
ALL_Test_Prediction=[]

"""## preprocessing"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preprocessing(df_,num_features):
    cols=df_.columns
    
    Train=df_.iloc[0:37830,:]
    Test=df_.iloc[37830-num_periods_input:,:]
    Train=Train.fillna(Train.mean())
    Test=Test.fillna(Test.mean())
    ################################################encoding########################
    
    Train=Train[['NDX','AAL','AAPL','ADBE','ADI','ADP','ADSK','AKAM','ALXN','AMAT','AMGN','AMZN',
                'ATVI','AVGO','BBBY','BIDU','BIIB','CA','CELG','CERN','CMCSA','COST',
                'CSCO','CSX','CTRP','CTSH','DISCA','DISH','DLTR','EA','EBAY','ESRX',
                'EXPE','FAST','FB','FOX','FOXA', 'GILD', 'GOOGL', 'INTC', 'JD', 'KHC', 'LBTYA',
                'LBTYK', 'LRCX', 'MAR', 'MAT', 'MCHP', 'MDLZ', 'MSFT', 'MU', 'MXIM', 'MYL', 'NCLH',
                'NFLX' ,'NTAP' ,'NVDA', 'NXPI' ,'PAYX' ,'PCAR', 'PYPL', 'QCOM', 'QVCA' ,'ROST',
                'SBUX', 'SIRI' ,'STX', 'SWKS', 'SYMC', 'TMUS', 'TRIP', 'TSCO', 'TSLA', 'TXN',
                'VIAB', 'VOD', 'VRTX', 'WBA', 'WDC', 'WFM', 'XLNX', 'YHOO']]
    
    Train=Train.values
    Train = Train.astype('float32')
    #################################################################################
    
    Test=Test[['NDX','AAL','AAPL','ADBE','ADI','ADP','ADSK','AKAM','ALXN','AMAT','AMGN','AMZN',
                'ATVI','AVGO','BBBY','BIDU','BIIB','CA','CELG','CERN','CMCSA','COST',
                'CSCO','CSX','CTRP','CTSH','DISCA','DISH','DLTR','EA','EBAY','ESRX',
                'EXPE','FAST','FB','FOX','FOXA', 'GILD', 'GOOGL', 'INTC', 'JD', 'KHC', 'LBTYA',
                'LBTYK', 'LRCX', 'MAR', 'MAT', 'MCHP', 'MDLZ', 'MSFT', 'MU', 'MXIM', 'MYL', 'NCLH',
                'NFLX', 'NTAP' ,'NVDA' ,'NXPI', 'PAYX' ,'PCAR' ,'PYPL' ,'QCOM' ,'QVCA' ,'ROST',
                'SBUX', 'SIRI' ,'STX', 'SWKS', 'SYMC', 'TMUS', 'TRIP', 'TSCO', 'TSLA', 'TXN',
                'VIAB', 'VOD', 'VRTX', 'WBA', 'WDC', 'WFM', 'XLNX', 'YHOO']]
    Test=Test.values
    Test = Test.astype('float32')
    #################################################################################
    Number_Of_Features=num_features
    split=num_periods_output+num_periods_input
    
    ####################### CUT THE PORTION of the data that we are working on 
        
    #############################  Normalization on train   #############
    
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
    #print('xxxxxx-------------',len(x_batches))
    x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)   
    y_batches=np.asarray(y_batches)
    #print('yyyy=======',len(y_batches))
    y_batches = y_batches.reshape(-1, num_periods_output, 1)   
    print('len x_batches ',len(x_batches))
    
    ###########################################TEST#####################################

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
    print('x---------',len(x_testbatches))

    x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features)
    y_testbatches=np.asarray(y_testbatches)
    #print('y=====',[len(s) for s in y_testbatches])
    y_testbatches = y_testbatches.reshape(len(y_testbatches), num_periods_output) 
    y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1) 
    print('len Test',len(Test))
    print('len xTestbatches',len(x_testbatches))
    
    return x_batches, y_batches, x_testbatches, y_testbatches

data_path1=r'/GBRT-for-TSF/Data/Multivariate/NASDAQ/nasdaq100.csv'
data_All=pd.DataFrame()
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]

range_list = [1]
data=pd.read_csv(data_path1,sep=',')

x_batches_Full, y_batches_Full,X_Test_Full,Y_Test_Full=preprocessing(data,82)
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


#print(len(All_Testing_Instances[0]))
#===========================calling MultiOutput XGoost=========================
All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances),len(All_Testing_Instances[0])))
Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full),num_periods_output))

#========== reshape train ==============================
All_Training_Instances=np.reshape(All_Training_Instances, (len(All_Training_Instances),len(All_Training_Instances[0])))
shuffled_batch_y=np.reshape(shuffled_batch_y, (len(shuffled_batch_y),num_periods_output))




print(All_Training_Instances.shape)
model=xgb.XGBRegressor(learning_rate =0.025,
 n_estimators=300,
 max_depth=2,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.9,
 colsample_bytree=0.9,
 scale_pos_weight=0.8,
 seed=27,silent=False)

multioutput=MultiOutputRegressor(model,n_jobs=-1).fit(All_Training_Instances,shuffled_batch_y)


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
