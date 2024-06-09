import os

import pandas as pd

from topic_clustering.utils.clusters import hdbscan, mini_batch_kmeans_optimized
from topic_clustering.utils.dimensional_reduction import DimensionalReduction
from topic_clustering.utils.folder import create_folder
from topic_clustering.utils.logger import get_logger

logger = get_logger(name="document_clusterer")


class DocumentClusterer:
    """A class for clustering documents using UMAP algorithm."""

    def __init__(
        self,
        config: dict,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
    ):
        """
        Initialize the DocumentClusterer class.

        Args:
            n_components (int, optional): The number of dimensions to reduce the data to. # noqa: E501
                Defaults to 2.
            n_neighbors (int, optional): The number of neighbors to consider for each data point. # noqa: E501
                Defaults to 15.
            min_dist (float, optional): The minimum distance between points \
                in the low-dimensional representation. Defaults to 0.1.
            metric (str, optional): The metric to use for distance computation. \
                Defaults to "cosine".
        """
        input_info = config.get("input_info")
        output_info = config.get("output_info")
        self.output_dir = output_info.get("output_dir", None)
        self.clustered_filename = output_info.get("clustered_filename", None)

        self.cluster_method = input_info.get("cluster_method", None)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = DimensionalReduction(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
        )

    def __fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        """
        Fit the UMAP model to the document embeddings \
            and transform them to the low-dimensional representation.

        Args:
            X (list[list[float]]): The document embeddings to be transformed.

        Returns:
            list[list[float]]: The transformed document embeddings \
                in the low-dimensional representation.
        """
        self.reducer.fit_transform(X)
        return self.reducer.fit_transform(X)

    def _save(self, result: pd.DataFrame):
        create_folder(self.output_dir)
        result.to_csv(
            os.path.join(self.output_dir, self.clustered_filename), index=False
        )

    def _clustering(self, X: list[list[float]]) -> list[int]:
        """
        Cluster the documents using HDBSCAN algorithm.

        Args:
            X (list[list[float]]): The document embeddings to be clustered.

        Returns:
            list[int]: The cluster labels for each document.
        """
        X_transform = self.__fit_transform(X)

        if self.cluster_method == "kmeans":
            cluster_labels, number_clusters = mini_batch_kmeans_optimized(X_transform)
            logger.info(f"Number of clusters: {number_clusters}")
        elif self.cluster_method == "hdbscan":
            cluster_labels, number_clusters = hdbscan(X_transform)
            logger.info(f"Number of clusters: {number_clusters}")
        else:
            raise ValueError("Invalid cluster method.")
        return cluster_labels

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster the documents in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the documents.

        Returns:
            pd.DataFrame: The DataFrame with the cluster labels.
        """
        df["cluster_labels"] = self._clustering(df["headline_embeddings"].tolist())
        self._save(df)
        return df
