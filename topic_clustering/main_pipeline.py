import pandas as pd

from topic_clustering.document_clusterer import DocumentClusterer
from topic_clustering.document_embedder import DocumentEmbedder
from topic_clustering.document_retriever import DataRetriever
from topic_clustering.utils.load_config import load_config
from topic_clustering.utils.logger import get_logger

logger = get_logger(name="main_pipeline")


def main_pipeline(config_file: str) -> pd.DataFrame:
    """
    Runs the main pipeline for topic clustering.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        pd.DataFrame: The clustered documents.
    """
    config = load_config(config_file)
    retriever = DataRetriever(config=config)
    df = retriever.get_data()

    embedder = DocumentEmbedder(config=config)
    df_embeddings = embedder.run(df=df)
    logger.info("Document embeddings have been successfully created.")

    clusterer = DocumentClusterer(config=config)
    df_clustering = clusterer.run(df_embeddings)
    logger.info("Documents have been successfully clustered.")

    return df_clustering
