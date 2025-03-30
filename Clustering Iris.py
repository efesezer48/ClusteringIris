import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# K-Means textbook implementation functions
def clustering_objective(data, clustering, reps):
    J_obj = 0
    for i in range(len(data)):
        J_obj += np.linalg.norm(data[i] - reps[int(clustering[i])])**2
    return J_obj / len(data)

def update_cluster(data, reps):
    new_clustering = np.zeros(len(data))
    for i in range(len(data)):
        distance = np.array([np.linalg.norm(data[i] - rep) for rep in reps])
        new_clustering[i] = np.argmin(distance)
    return new_clustering

def update_reps(data, clustering):
    new_reps = []
    for j in range(int(max(clustering) + 1)):
        cluster_points = data[clustering == j]
        if len(cluster_points) > 0:
            new_reps.append(np.mean(cluster_points, axis=0))
    return np.array(new_reps)

def Kmeans_alg(data, reps):
    clustering = update_cluster(data, reps)
    J_obj = [clustering_objective(data, clustering, reps)]
    
    while True:
        new_reps = update_reps(data, clustering)
        clustering = update_cluster(data, new_reps)
        J_obj.append(clustering_objective(data, clustering, new_reps))
        if np.allclose(new_reps, reps):
            break
        reps = new_reps
    
    return new_reps, clustering, J_obj

# Initialize with three random points as cluster centers
np.random.seed(42)
random_indices = np.random.choice(len(data), 3, replace=False)
z_init = data[random_indices]

# Run textbook K-Means
final_reps, final_clusters, final_obj = Kmeans_alg(data, z_init)

# Run Sklearn K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
sklearn_clusters = kmeans.fit_predict(data)

# Compare clustering with actual labels
print("Cluster assignment (Textbook K-Means):")
print(final_clusters)
print("Cluster assignment (Sklearn K-Means):")
print(sklearn_clusters)

# Visualization of clustering
plt.figure(figsize=(12, 5))

# Textbook K-Means Clustering Plot
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=final_clusters, cmap='viridis', edgecolor='k')
plt.scatter(final_reps[:, 0], final_reps[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Textbook K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Sklearn K-Means Clustering Plot
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=sklearn_clusters, cmap='viridis', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Sklearn K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()
