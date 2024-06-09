from typing import Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def mini_batch_kmeans_optimized(
    data: np.ndarray, max_clusters: int = 22, batch_size: int = 100, max_iter: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Perform optimized Mini-Batch K-Means clustering on the given data.

    Parameters:
        data (np.ndarray): The input data for clustering.
        max_clusters (int): The maximum number of clusters to consider \
            (default: 22).
        batch_size (int): The number of samples to use in each mini-batch \
            (default: 100).
        max_iter (int): The maximum number of iterations for each mini-batch \
            (default: 50).

    Returns:s
        Tuple[np.ndarray, int]: A tuple containing the cluster labels \
            and the best number of clusters.
    """
    # Calculate the silhouette score
    scores = []
    best_score = -1
    best_cluster = 0
    labels = None
    for n in tqdm(range(2, max_clusters), desc="Optimizing KMeans"):
        kmeans = MiniBatchKMeans(n_clusters=n, batch_size=batch_size, max_iter=max_iter)
        kmeans.fit(data)
        curr_labels = kmeans.predict(data)
        score = silhouette_score(data, curr_labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_cluster = n
            labels = curr_labels

    return labels, best_cluster


def hdbscan(
    data: np.ndarray, min_cluster_size: int = 5, min_samples: int = 5
) -> Tuple[np.ndarray, int]:
    """
    Perform HDBSCAN clustering on the given data.

    Parameters:
        data (np.ndarray): The input data for clustering.
        min_cluster_size (int): The minimum number of samples in a cluster \
            (default: 5).
        min_samples (int): The minimum number of samples required \
            to form a dense region (default: 5).

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the cluster labels and \
            the number of clusters.
    """
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdbscan.fit_predict(data)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, num_clusters
