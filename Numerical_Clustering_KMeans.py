import pandas as pd 
import datetime as dt 
import warnings 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans 
import numpy as np 
from IPython.display import display
import matplotlib.pyplot as plt

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
pd.plotting.scatter_matrix(dataset, alpha = 0.3, figsize = (14,8), diagonal = 'kde'); 
plt.show(block = False)
pca = PCA().fit(dataset) # Generate PCA results plot 
 
# applying PCA by fitting the dataset with dimensions that account for over 75% of variance 
pca = PCA(n_components=2).fit(dataset) 
 
# Transform the dataset using the PCA fit above 
customer_data = pca.transform(dataset) 
 
# Create a DataFrame for the data 
reduced_data = pd.DataFrame(customer_data, columns = ['Dimension 1', 'Dimension 2']) 
 
#set number of clusters to few different values for checking silhoutte score of Kmeans 
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
cluster_sil = [] 
for n in n_clusters: 
     
    # Apply KMeans algorithm to the dataset 
    clusterer = KMeans(n_clusters=n).fit(reduced_data) 
 
    # Predict the cluster for each data point 
    preds = clusterer.predict(reduced_data) 
 
    # Calculate the mean silhouette coefficient for the number of clusters chosen     
    score = silhouette_score(reduced_data,preds)
    cluster_sil.append(score) 

plt.figure()
plt.plot(n_clusters, cluster_sil)
plt.title("Silhouette Scores for different Cluster Sizes")
plt.ylabel("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.show(block = False)     
#calculate Elbow point     
wcss=[] 
for i in range(1,10): 
    kmeans=KMeans(n_clusters=i,init='k-means++',)     
    kmeans.fit(reduced_data)     
    wcss.append(kmeans.inertia_) 
plt.figure()
plt.plot(range(1,10),wcss) 
plt.title('Elbow Method') 
plt.xlabel('No. of cluster') 
plt.ylabel('wcss: sum of dist. of datapoint to their closest cluster center' )
plt.show(block=False) 

#clustering datapoints using K-means algorithm 
#Set the number of clusters based on elbow point 
k=5 
kmeans_customers=KMeans(n_clusters=k) 
X=reduced_data.values 
kmeans_customers.fit(X) 
cluster_pred=kmeans_customers.predict(X) 
cluster_centers=kmeans_customers.cluster_centers_ 
colors=['red','blue','green', 'cyan','magenta'] 
# Visualising the clusters 
for i in range(0,k):
    plt.figure()     
    plt.scatter(X[cluster_pred==i,0],X[cluster_pred==i,1], s = 100, c = colors[i], label ='cluster '+str(i+1)) 
    plt.scatter(cluster_centers[:,0],cluster_centers[:,1], s = 300, c = 'yellow', label = 'Centroids') 
    plt.title('Customer Segmentation') 
    plt.xlabel('Dimension 1') 
    plt.ylabel('Dimension 2') 
    plt.legend() 
    plt.show(block=False) 
 
# Inverse transform the centers 
actual_centers = pca.inverse_transform(cluster_centers) 
 
# Display the actual centers 
segments = ['Segment {}'.format(i) for i in range(0,len(cluster_centers))] 
actual_centers = pd.DataFrame(np.round(actual_centers), columns = dataset.keys()) 
actual_centers.index = segments 
display(actual_centers) 
plt.show()
