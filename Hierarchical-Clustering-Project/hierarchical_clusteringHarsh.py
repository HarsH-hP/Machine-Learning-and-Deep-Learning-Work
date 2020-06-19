# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:48:11 2020

@author: harsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
X = df.iloc[:, [3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
# Looking at the plot we can figure out optimal number of clusters required to build the model

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred = ac.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c = 'red', s = 100, label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c = 'green', s = 100, label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c = 'blue', s = 100, label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c = 'cyan', s = 100, label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c = 'magenta', s = 100, label = 'Cluster 5')
plt.title('Hierarchical Clustering')
plt.xlabel('Income')
plt.ylabel('Spend')
plt.legend()
plt.show()