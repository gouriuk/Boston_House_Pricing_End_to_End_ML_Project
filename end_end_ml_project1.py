# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:42:10 2023

@author: Gouri
"""
#Linear Regression ML implementation:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn

#let's load the Boston House Pricing dataset:
from sklearn.datasets import load_boston
boston=load_boston()
type(boston)
boston.keys()

#Checking the description of dataset:
print(boston.DESCR)

boston.data
boston.feature_names
boston.target
#Preparing the dataset:
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df.head()
#Creating output or dependent feature:
df['Price']=boston.target
df.head()
#Analysing the dataset:
df.info()
df.describe()

#finding the missing values:
df.isnull().sum()

#Exploratory data analysis:
#Finding correlation btn the features:
df.corr()
#more positive values indicate they are more positively correlated to dependent feature..
#more negative values indicate they are more negatively correlated to dependent feature
#if its zero..they are highly correlated..
#here we can find correation btn the independent features and also the correlation btwn
#independent and dependent features.
#We have to check for multicollinearity btn independent features..i.e if they are highly 
#correlated then we can drop either of them to increase model's performance..
import seaborn as sns
sns.pairplot(df)

#if we want to analyze correlation btn CRIM and Price features:
plt.scatter(df['CRIM'],df['Price'])
plt.xlabel('Crime Rate')
plt.ylabel('Price of house')
#As Crime rates increase prices of house decrease..these two are inversely correlated..

plt.scatter(df['RM'],df['Price'])
plt.xlabel('Avg no. of rooms per dwelling')
plt.ylabel('Price of house')
sns.regplot(x='RM',y='Price',data=df)

plt.scatter(df['LSTAT'],df['Price'])
plt.xlabel('% lower status of population')
plt.ylabel('Price of house')
sns.regplot(x='LSTAT',y='Price',data=df)

plt.scatter(x='CHAS', y='Price', data=df)
plt.xlabel('Charles River dummy variable')
plt.ylabel('Price')
sns.regplot(df['CHAS'],df['Price'])
#these two features are not at all correlated..there is no linearity btn these two features..
#there must be either inverse or direct correlation btn features to build regression model
plt.scatter(df['PTRATIO'],df['Price'])
sns.regplot(df['PTRATIO'],df['Price'])
#there exists inverse correlation btn PTRATIO and Price

#Creating model:
#Dividing dataset into Independent and Dependent features:
X=df.iloc[:,:-1]
X.head()
y=df.iloc[:,-1]
y.head()

#Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Standardization:When we use linear regression, we achieve Gradient descent at the earliest 
# if we standardize all the features with same units..we do it with the help of 
#StandardScaler():
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_train
X_test=sc.transform(X_test) #we use transform() function just to apply different techniques for 
# Test data..so that our model doesn't get to know about test data..  
#we use internally Gradient Descent and our main aim is to reach Global minima, if
#we standardize all units to the same, convergence to global minima happens quickly..

#pickling the standardization:
import pickle
pickle.dump(sc,open(r'G:\revision_ml\end_to_end_ml_projects\Boston_House_Pricing_End_to_End_ML_Project\scaling.pkl','wb'))    
#Model training:
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
#Now the model is created..and we are creating a hyperplane as we are having so many features in it..
#when we create regression model, we have two things..coefficients and intercept:
print(reg.coef_)
#these coefficients are equal to no. of independent features..
#for unit increase in CRIM value there is 1.1083 decrease in price value
print(reg.intercept_)

#On which parameters model has been trained:
reg.get_params()

#Once the model is trained, we have to do prediction:
#Prediction with Test data:
reg_pred=reg.predict(X_test)
reg_pred
#To check model has predicted correctly or not..we have to do it by plotting scatter plot:
#Plot a scatter plot:
plt.scatter(y_test,reg_pred)
#there exists linearity in model's prediction, which indicates there exists 
#correlation btwn predicted values and true value i.e y_test values and hence model has performed well.. 

#Plotting w.r.t error or residual:
residual=y_test-reg_pred
sns.displot(residual,kind='kde')
#the graph is normally distributed but there exist outliers towards right side of the graph and the gap is more
#Assumption of linear regression model:
#we should get normal distribution for residuals
#but here we get some outliers...max errors or differnece lies btn -10 to 10 and 
#and some other differnce lies btn 15 to 30..

#Scatter plot w.r.t prediction and residuals:
plt.scatter(reg_pred,residual)
#this plot has uniform distribution of points...datapoints are scattered uniformly all over the graph

#Performance metrics:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test, reg_pred)) 
print(np.sqrt(mean_squared_error(y_test, reg_pred)))

#R Square and Adjusted R Square:
#Formula:
#R^2=1-(SSR/SST)
#Adjusted R^2=1-[(1-R^2)*(n-1)/(n-k-1)]
#n:no. of observations, k=no. of predictor variables
from sklearn.metrics import r2_score
score=r2_score(y_test, reg_pred)
score
#this score is good..but more towards 1 more better will be the score

#Adjusted R^2:
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
#Adjusted R^2 is less than R^2..which is the actual property that should happen..

#New data prediction with regression model, when new data is in bulk:
boston.data[0].shape
boston.data[0].reshape(1,-1).shape
#transformation of new data:
sc.transform(boston.data[0].reshape(1,-1))    
reg.predict(sc.transform(boston.data[0].reshape(1,-1)))

#Pickling the model file for deployment:
import pickle

#to convert model file into pickle file:
pickle.dump(reg,open(r'G:\revision_ml\ML\regression_model.pkl','wb'))
#this pickle file is serialised format file, so that it can be deployed in any server,
# if we want to do prediction from any web application we can do it.
pickled_model=pickle.load(open(r'G:\revision_ml\ML\regression_model.pkl','rb'))

#Prediction:
pickled_model.predict(sc.transform(boston.data[0].reshape(1,-1)))
