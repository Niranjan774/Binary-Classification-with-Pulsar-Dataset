

import pandas as pd
import numpy as np
import pickle

train  = pd.read_csv('train.csv')

train.drop('id',axis=1,inplace=True)
train['Class'] = train['Class'].fillna(0)
train['SD_DMSNR_Curve'] = train['SD_DMSNR_Curve'].fillna(0)
train['EK_DMSNR_Curve'] = train['EK_DMSNR_Curve'].fillna(0)
train['Skewness_DMSNR_Curve'] = train['Skewness_DMSNR_Curve'].fillna(0)

x_train = train.iloc[:,0:-1].values

y_train = train.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)
lr.predict([[192,56,1,-1,76,33,1,-5]])[0]

pickle.dump(lr ,open('model.pkl', 'wb'))