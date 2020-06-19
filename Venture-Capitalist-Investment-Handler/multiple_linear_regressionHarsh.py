# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:23:48 2020

@author: harsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values

# Working on Categorical Variable into dummy variables because(There are three diff values of categories)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
X[:,3] = label.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features = [3])
X = onehot.fit_transform(X).toarray()

"""
                # Another way of Encoding categorical data 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

"""

# Avoiding the Dummy variable trap
X = X[: , 1:]

# Seperating into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Building Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)

# Predicting using the model
predict = linear.predict(X_test)
print(linear.score(X_test,Y_test)) #Accuracy is 0.93470


#Plotting the graph to look data

plt.scatter(X_train[:,2], Y_train, color = 'red')
plt.title("UnOptimized Model (Train set)")
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.show()    

plt.scatter(X_train[:,3], Y_train, color = 'red')
plt.title("UnOptimized Model (Train set)")
plt.xlabel("Administration")
plt.ylabel("Profit")
plt.show() 

plt.scatter(X_train[:,4], Y_train, color = 'red')
plt.title("UnOptimized Model (Train set)")
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.show() 

# Backward Elimination method to solve the problem of various dummy variables and get statistically best variable
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 )

    # Assuming Significance Level as 0.05 so remove all columns whose p value is >0.05
X_opt = X[ : ,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()     # Ordinary Least Square model
regressor_OLS.summary()

X_opt = X[ : , [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[ : , [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[ : , [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[ : , [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

   # From this we can see that the most suitable statistically variable is column 3 i.e R&D column
    
    
# Using X_opt splitting the Training and test data set
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_opt,Y, test_size = 0.2, random_state = 0)

#Building the model again with optimized dataset
regressor = LinearRegression()
regressor.fit(X1_train,Y1_train)
y1_predict = regressor.predict(X1_test)

print(regressor.score(X1_test,Y1_test))    #Accuracy is 0.94645
    

#Plotting the result

plt.scatter(X1_train[:,1], Y1_train, color = 'red')
plt.plot(X1_train[:,1], regressor.predict(X1_train), color = 'blue')
plt.title("Optimized Model (Train set)")
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.show()    

plt.scatter(X1_test[:,1], Y1_test, color = 'red')
plt.plot(X1_test[:,1], y1_predict, color = 'blue')
plt.title("Optimized Model (Test set)")
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.show()    
