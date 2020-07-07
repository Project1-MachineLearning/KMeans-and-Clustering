import pandas as pd  # To use CSV

import matplotlib.pyplot as plt  # plots

customers_data = pd.read_csv("Customers.csv")  # reading data from Csv

customers_data = customers_data.drop('CustomerID', axis=1)  # removing the column from the dataset
customers_data = customers_data.drop('Gender', axis=1)  # removing the column from the dataset
customers_data = customers_data.drop('Age', axis=1)  # removing the column from the dataset

customers_data.columns  # print column names
print(customers_data.head())  # Prints Top 10 rows

from sklearn.decomposition import PCA  # Dimensionality reduction
pca_reducer = PCA(n_components=2)  # Dimensionality labeling whole columns into 2
reduced_data = pca_reducer.fit_transform(customers_data)  # reducing the data with PCA
print(reduced_data.shape)  # print reduced shape

from sklearn.cluster import KMeans  # Clustering using Kmeans library
km = KMeans(n_clusters=5)  # setting number of clusters to 5 by default
cluster = km.fit(reduced_data)  # Giving data to Kmeans clustering model
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Datapoints')  # Considering scatter plot on PCA1 and PCA2
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], label='Clusters')  # centers of clusters
plt.title("Sklearn version of KMeans with PCA")  # title for plot
plt.legend()
plt.show()

# predict the cluster for each data point
y_cluster_kmeans = km.predict(reduced_data)  # predict the cluster for each row
from sklearn import metrics
score = metrics.silhouette_score(reduced_data, y_cluster_kmeans)  # getting silhoutte score - indicates how well the data is close/similar to the cluster
print("Silhoutte Score: " + str(score))  # print the score -> ranges from -1 to 1 -> positive means the data is more similar to the cluster

# ##elbow method to know the number of clusters
wcss = [] # Model inertia ->
for i in range(1, 11):  # iterating from i value 1 to 11
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)  # does k-means clustering by setting number of clusters to i
    kmeans.fit(reduced_data)  # after setting number of clusters -> fit the data to the model
    wcss.append(kmeans.inertia_)  # getting note of the error for each iteration


# Plotting the sum of squared errors that is received from the elbow method
plt.plot(range(1, 11), wcss)
plt.title('the elbow method with PCA')  # setting title for the graph
plt.xlabel('Number of Clusters')  # setting the x label
plt.ylabel('Wcss')  # setting the y label for the graph
plt.show()  # Finally visualize the graph


