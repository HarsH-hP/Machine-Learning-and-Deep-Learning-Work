# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:38:03 2020

@author: harsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

                        # Cleaning the texts


                        # Stemming Process
# Here all the words are converted into root word
                       # eg:- { loved - past tense}
                       #      { love - root word}
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):    
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])    # Removing everything except the alphabets
    review = review.lower()     # Converting into lowercase
    review = review.split()     # Splitting the sentence by space and forms a list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('English'))] # set is used because set algorithm is faster than the list algorithm
    review = ' '.join(review)   # Joining the words using space
    corpus.append(review)

                # Creating the Bag of Words Model
# Take all the diff words of 1000 reviews here 
# without taking duplicates or triplicates which is unique and will create column of each unique word    
# Put all these columns in a table where the rows correspond to reviews
# This will create a sparse matrix(means matrix contains 0 value a lot)
# Bag of Words model simplify all the reviews and try to minimize the number of words and also creating sparse matrix through tokenization
                
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()  
Y = df.iloc[:, 1].values          
            
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train) 
y_pred = nb.predict(X_test)
print(nb.score(X_test,Y_test))            

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)            

            