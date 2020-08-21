# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import Imputer
train=pd.read_csv('train.csv',delimiter=',')
print(train)

#dropping the unnecessary data
x=train.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'])
print(x)

#convertimg dataset into an array of independent variables
X=x.iloc[:,1:].values
Y=x.iloc[:,0].values
#dealing with NaN
imputer=Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer=imputer.fit(X[:,2:5])
X[:,2:5]=imputer.transform(X[:,2:5])
print(X)

#encoded categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,1]=le1.fit_transform(X[:,1])
print(X)

#scaling the data to more concise data using mean and std deviation
from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
X[:,:]=sd.fit_transform(X[:,:])
print(X)

#building ANN
import keras
from keras.models import Sequential 
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=5, kernel_initializer='uniform',activation='relu',input_dim=6))
model.add(Dense(units=4, kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

#compiling the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X,Y,batch_size=10,epochs=100)

#compiling test data

test=pd.read_csv('test.csv',delimiter=',')
t=test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'])

#dealing with NaN in test data
avg=t['Age'].mean()
t=t.fillna(avg)
T=t.iloc[:,:].values

#encoding Male-Female
le1 = LabelEncoder()
T[:,1]=le1.fit_transform(T[:,1])
print(T)

#scaling the data
T[:,:]=sd.fit_transform(T[:,:])
print(T)

#getting predictions of test data

prediction = model.predict(T).tolist()
se = pd.Series(prediction)
output=test.iloc[:,:1]
output['check'] = se
output['check'] = output['check'].str.get(0)
series = []
for val in output.check:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
output['Survived'] = series
o=output.drop(columns=['check'])
print(o)

# saving the dataframe 
o.to_csv('submission1.csv')