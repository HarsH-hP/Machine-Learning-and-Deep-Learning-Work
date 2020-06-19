# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:04:04 2020

@author: harsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
X = df.iloc[:, [3,4]].values

# Finding the optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    cluster = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    cluster.fit(X)
    wcss.append(cluster.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('Sum for each Clusters')
plt.show()

# Looking at the plot we can see that the elbow method finds the value 5 on X-axis. So no. of clusters = 5
cluster = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)    
y_pred = cluster.fit_predict(X)

# Visualizing the Clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c = 'red', s = 100, label = 'Cluster 1') #High_Income/Low Spend
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c = 'green', s = 100, label = 'Cluster 2') # Medium/Medium
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c = 'blue', s = 100, label = 'Cluster 3') # High/High
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c = 'cyan', s = 100, label = 'Cluster 4') # Low/High
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c = 'magenta', s = 100, label = 'Cluster 5') # Low/Low
plt.scatter(cluster.cluster_centers_[:,0], cluster.cluster_centers_[:,1], c = 'orange', s = 300, label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
