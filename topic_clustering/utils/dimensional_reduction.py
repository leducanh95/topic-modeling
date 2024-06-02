import numpy as np
import umap.umap_ as umap


class DimensionalReduction:
    """Class for performing dimensional reduction using UMAP algorithm."""

    def __init__(
        self, n_components: int, n_neighbors: int, min_dist: float, metric: str
    ):
        """
        Initialize the DimensionalReduction class.

        Args:
            n_components (int): The number of dimensions to reduce the data to.
            n_neighbors (int): The number of neighbors to consider for each data point.
            min_dist (float): The minimum distance between points \
                in the low-dimensional representation.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the UMAP model to the data and transform it \
            to the low-dimensional representation.

        Args:
            X (np.ndarray): The input data to be transformed.

        Returns:
            np.ndarray: The transformed data in the low-dimensional representation.
        """
        return self.reducer.fit_transform(X)
