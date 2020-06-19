# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:21:34 2020

@author: harsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

records = []
for i in range(0,7501):
    records.append([str(df.values[i,j]) for j in range(0,20)])
        # Taking the rowwise list into the records which will be used to build the model 

# Training Apriori on the dataset
from apyori import apriori   
# Here we want an item bought atleast 3 times a day so 3*7 a week min_support=(3*7)/7500 = 0.0028 = 0.003 approx
# min_confidence ------here 20% is taken which means 20% of the time the rule is correct
# min_lift ----------- default value 3 is taken so that we can get some good rules. 
rules = apriori(records, min_support = 0.003, min_confidence = 0.2 , min_lift = 3, min_length = 2) 
    
            # Visualizing the result
results = list(rules)
print(results)
            # Rules obtained by apriori model is already sorted by their relevance. 
            #Relevance is combination of supprt, confidence and lift
for i in range(0,len(results)):
    print("Rule-" + str(i) +" : "+ str(results[i][0]) + "\nSupport = " + str(results[i][1]) )
    print("Item A : " + str(results[i][2][0][0]) + "\nItem B : " + str(results[i][2][0][1]) )
    print("Confidence = " + str(results[i][2][0][2]) + "\nLift = " + str(results[i][2][0][3]) + "\n")