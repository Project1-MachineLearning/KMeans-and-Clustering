# Without dimensionality Reduction

import pandas as pd  # for importing CSV file
import matplotlib.pyplot as plt  # for Plotting
customers_data = pd.read_csv("Customers.csv")  # Import the CSV file
customers_data = customers_data.drop('CustomerID', axis=1)  # dropping column
customers_data = customers_data.drop('Gender', axis=1)  # dropping Column
customers_data = customers_data.drop('Age', axis=1)  # dropping Column

customers_data.columns  # Printing Column names
print(customers_data.head())  # printing top 10 columns


from sklearn.cluster import KMeans  # Clustering
km = KMeans(n_clusters=5)  # set clusters
cluster = km.fit(customers_data)  # input initial data
plt.scatter(customers_data.iloc[:, 0], customers_data.iloc[:, 1], label='Datapoints')  # Considering scatter plot on PCA1 and PCA2
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], label='Clusters')  # centers of clusters
plt.title("Sklearn version of KMeans without PCA")  # title for plot
plt.legend()
plt.show()

# predict the cluster for each data point
y_cluster_kmeans = km.predict(customers_data)  # predit the cluster for each row
from sklearn import metrics
score = metrics.silhouette_score(customers_data, y_cluster_kmeans) # getting silhoutte score - indicates how well the data is close/similar to the cluster
print("Silhoutte Score: " + str(score)) # print the score -> ranges from -1 to 1 -> positive means the data is more similar to the cluster

# ##elbow method to know the number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) # does k-means clustering by setting number of clusters to i
    kmeans.fit(customers_data)  # after setting number of clusters -> fit the data to the model
    wcss.append(kmeans.inertia_)  # getting note of the error for each iteration

# Plotting the sum of squared errors that is received from the elbow method
plt.plot(range(1, 11), wcss)
plt.title('the elbow method without PCA')  # setting title for the graph
plt.xlabel('Number of Clusters')  # setting the x label
plt.ylabel('Wcss')  # setting the y label for the graph
plt.show()  # Finally visualize the graph

