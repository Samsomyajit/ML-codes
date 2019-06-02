# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:36:01 2019

@author: Sam

"""
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')

#Dividing Dataset

X= dataset.iloc[:,2:-1].values
Y= dataset.iloc[:,-1].values

#missing values fix

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#encoding categorical data
"""
from sklearn.preprocessing import LabelEncoder 
lb_X= LabelEncoder()
X[:,0] =lb_X.fit_transform(X[:,0])


oneh = OneHotEncoder(categorical_features=[0])
X= oneh.fit_transform(X).toarray()

lb_Y = LabelEncoder()
Y = lb_Y.fit_transform(Y)
"""
#train test split

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)


#predict

Y_pred= classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,Y_pred)






