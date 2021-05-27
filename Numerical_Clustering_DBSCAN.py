import pandas as pd 
import datetime as dt 
import warnings 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans 
import numpy as np 
from IPython.display import display
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.cluster import DBSCAN 

# set filepath 
datasourcepath='res/Mall_Customers.csv' 
#load dataset 
dataset=pd.read_csv(datasourcepath) 
dataset.info() 
#Restrict dataframe to annual income greater than 0 
dataset = dataset[(dataset['Annual Income (k$)']>0)] 
#Remove gender 
dataset.pop('Gender') 
#Remove customerID 
dataset.pop('CustomerID') 
#display information about the dataset 
dataset.info() 
 
outliers  = [] 
# For each feature find the data points are either too high or too low 
for feature in dataset.keys(): 
     
    # Calculate Q1 (25th percentile of the data) for the feature 
    Q1 = np.percentile(dataset[feature],25) 
     
    # Calculate Q3 (75th percentile of the data) for the feature 
    Q3 = np.percentile(dataset[feature],75) 
     
# set outlier step to 1.5 times the interquartile range     
    step = (Q3-Q1) * 1.5         
# Display the outliers 
    print("Possible data point outliers for the feature '{}':".format(feature))     
    out = dataset[~((dataset[feature] >= Q1 - step) & (dataset[feature] <= Q3 + step))]     
    display(out)     
    outliers = outliers + list(out.index.values) 
     
#list of more outliers which are the same for multiple features. 
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))  
 
# create a scatter matrix for each pair of features in the dataset 
pd.plotting.scatter_matrix(dataset, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show(block = False) 
pca = PCA().fit(dataset) # Generate PCA results plot 
 
# applying PCA by fitting the dataset with dimensions that account for over 75% of variance 
pca = PCA(n_components=2).fit(dataset) 
 
# Transform the dataset using the PCA fit above 
customer_data = pca.transform(dataset) 
 
# Create a DataFrame for the data 
reduced_data = pd.DataFrame(customer_data, columns = ['Dimension 1', 'Dimension 2']) 
 
#clustering datapoints using dbscan algorithm 
dbscan = DBSCAN(eps=9.25,min_samples=8) 
 
# Fitting the pca transformed data 
dbscan.fit(customer_data) 
 
#get unique label unique values and count 
print(Counter(dbscan.labels_).keys() ) 
print(Counter(dbscan.labels_).values() ) 
score = silhouette_score(customer_data,dbscan.labels_) 
print("The silhouette_score for {} clusters is {}".format(5,score))  
#Visualize clustering 
plt.figure()
for i in range(2, customer_data.shape[0]):     
    if dbscan.labels_[i] == 0: 
        c1 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='r', marker='+') 
    elif dbscan.labels_[i] == 1: 
        c2 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] ==2: 
        c3 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='y', marker='p')
    elif dbscan.labels_[i] ==3: 
        c4 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='m', marker='p')
    elif dbscan.labels_[i] ==4: 
        c5 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='c', marker='p')
    elif dbscan.labels_[i] == -1:         
        c6 = plt.scatter(customer_data[i, 0], customer_data[i, 1], c='b', marker='*')
 
plt.legend([c1, c2, c3,c4,c5,c6], ['Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4','Cluster 5','Noise']) 
plt.title('DBSCAN finds 5 clusters and noise') 
plt.xlabel('Dimension 1') 
plt.ylabel('Dimension 2') 
plt.show()
