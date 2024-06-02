# from hdbscan import HDBSCAN
# from sklearn.cluster import DBSCAN
from utils.dimensional_reduction import DimensionalReduction


class DocumentClusterer:
    """A class for clustering documents using UMAP algorithm."""

    def __init__(
        self,
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

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
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
