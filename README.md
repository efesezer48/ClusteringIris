# Iris Dataset Clustering with K-Means

This project demonstrates the application of K-Means clustering on the Iris dataset using both a **manual (textbook)** implementation and **Scikit-learn's built-in KMeans**. It visually compares the clustering results of both methods.

# 📊 Dataset
- **Iris dataset** (150 samples, 4 features)
- No labels used during clustering — purely unsupervised.

# 🧠 Implementations
- **Textbook K-Means**:
  - Manual calculation of cluster assignments and centroid updates
  - Tracks convergence via objective function (within-cluster sum of squares)

- **Scikit-learn KMeans**:
  - `sklearn.cluster.KMeans` with 3 clusters for comparison

# 📈 Visualization
- Side-by-side scatter plots of:
  - Textbook K-Means result
  - Sklearn K-Means result
- Centroids marked with red **X**

# 🚀 How to Run
Run on Visual Studio Code
or
```bash
python Clustering\ Iris.py
