import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# load dataset from CSV file
file_path = r"D:\ProjectSource\K-means\IRIS.csv"
iris_data = pd.read_csv(file_path)

# Extract features and true labels
data = iris_data.iloc[:, :-1].values  # last column is species
target = pd.factorize(iris_data.iloc[:, -1])[0]  # encode species names as numeric labels
species = iris_data.iloc[:, -1].unique()  # get unique species names

# standardize the dataset for better clustering performance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# apply K-Means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=43)
kmeans.fit(data_scaled)
# cluster assignments
labels = kmeans.labels_

# map predicted clusters to species based on majority overlap
cluster_to_species = {}
for cluster in np.unique(labels):
    # Find the most common true label in this cluster
    true_labels = target[labels == cluster]
    mapped_species = species[np.bincount(true_labels).argmax()]
    cluster_to_species[cluster] = mapped_species

# visualize the clusters in 2D using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    cluster_points = data_pca[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"{cluster_to_species[cluster]}")

centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset (PCA Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

