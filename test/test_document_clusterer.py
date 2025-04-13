import pytest

from topic_clustering.document_clusterer import (
    DocumentClusterer,  # Adjust import based on actual implementation
)


def test_cluster_documents():
    """
    Test the `cluster_documents` method of the `DocumentClusterer` class.
    This test verifies that the `cluster_documents` method correctly clusters
    a list of documents. It checks the following:
    - The returned value is a list.
    - The list contains at least one cluster.
    Example:
        assert isinstance(clusters, list)
        assert len(clusters) > 0
    """
    clusterer = DocumentClusterer()
    documents = ["Document 1 text", "Document 2 text", "Document 3 text"]
    clusters = clusterer.cluster_documents(documents)

    assert isinstance(clusters, list), "Clusters should be a list"
    assert len(clusters) > 0, "There should be at least one cluster"


def test_empty_documents():
    """
    Test the behavior of the DocumentClusterer with an empty list of documents.
    This test ensures that when an empty list of documents is provided as input,
    the cluster_documents method returns an empty list of clusters.
    Assertions:
        - The returned clusters should be an empty list.
    """
    clusterer = DocumentClusterer()
    documents = []
    clusters = clusterer.cluster_documents(documents)

    assert clusters == [], "Clusters should be empty for empty input"


def test_invalid_input():
    """
    Test the DocumentClusterer for handling invalid input.
    This test ensures that the `cluster_documents` method raises a ValueError
    when provided with invalid input (e.g., None).
    """
    clusterer = DocumentClusterer()
    with pytest.raises(ValueError):
        clusterer.cluster_documents(None)  # Assuming None is invalid input
