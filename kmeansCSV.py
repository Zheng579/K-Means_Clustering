# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# load data from CSV -- https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download
file_path = r"D:\ProjectSource\K-means\IRIS.csv"
iris_data = pd.read_csv(file_path)

# extract feature (exclude target column 'species')
features = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# standardized features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# apply k-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features_scaled)
cluster_labels = kmeans.labels_

# evaluate the clustering performance with Silhouette Score
silhouette_avg = silhouette_score(features_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# map cluster to iris name
# map the clusters manually if needed
cluster_mapping = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

# replace numeric cluster labels with names
iris_data['Cluster'] = [cluster_mapping[label] for label in cluster_labels]

# visualized clusters in 2D using Principal component analysis (PCA)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# define cluster color
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
cluster_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#plot using matplotlib
plt.figure(figsize=(8, 6))
for i, color in enumerate(colors):
    cluster_points = features_pca[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=cluster_names[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset (PCA Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()


