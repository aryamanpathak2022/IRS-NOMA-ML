# kgmm_clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# ---------- Normalize CSI vectors or positions ----------
def normalize_vectors(vectors):
    return normalize(vectors, norm='l2')

# ---------- K-GMM Clustering ----------
def k_gmm_clustering(positions, n_clusters=5):
    """
    Clusters users into n_clusters using KMeans initialized GMM.
    Input: positions or CSI vectors (num_users, 2 or more)
    Output: cluster_labels, cluster_centers
    """
    normed = normalize_vectors(positions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normed)
    gmm = GaussianMixture(n_components=n_clusters, means_init=kmeans.cluster_centers_, random_state=0)
    gmm.fit(normed)
    labels = gmm.predict(normed)
    centers = gmm.means_
    return labels, centers

# ---------- Cluster Plotting ----------
def plot_clusters(positions, labels, centers=None, title="User Clustering via K-GMM"):
    n_clusters = np.unique(labels).shape[0]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow']
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        cluster_points = positions[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', color=colors[i % len(colors)])
        if centers is not None:
            plt.scatter(centers[i][0], centers[i][1], color='black', s=120, marker='X', edgecolors='white')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()