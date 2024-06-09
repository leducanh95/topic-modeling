import os

import pandas as pd
from sentence_transformers import SentenceTransformer

from topic_clustering.utils.folder import create_folder


class DocumentEmbedder:
    """A class for encoding documents into embeddings \
        using SentenceTransformer model."""

    def __init__(self, config: dict):
        """Initializes the DocumentEmbedder class."""
        self.model = self.__init_model(config)
        output_info = config.get("output_info")
        self.output_dir = output_info.get("output_dir", None)
        self.processed_filename = output_info.get("processed_filename", None)

    def __init_model(self, config) -> SentenceTransformer:
        """
        Initializes the SentenceTransformer model.

        Returns:
            SentenceTransformer: The initialized SentenceTransformer model.
        """
        input_info = config.get("input_info")
        embedding_model = input_info.get("embedding_model", None)
        model = SentenceTransformer(embedding_model)
        return model

    def _save(self, result: pd.DataFrame):
        create_folder(self.output_dir)
        result.to_csv(
            os.path.join(self.output_dir, self.processed_filename), index=False
        )

    def get_embedding(self, documents: list[str]) -> list[list[float]]:
        """
        Encodes the given documents into embeddings.

        Args:
            documents (list[str]): The list of documents to be encoded.

        Returns:
            list[list[float]]: The encoded document embeddings.
        """
        document_embedding = self.model.encode(documents)
        return document_embedding.tolist()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the documents in the DataFrame into embeddings.

        Args:
            df (pd.DataFrame): The DataFrame containing the documents.

        Returns:
            pd.DataFrame: The DataFrame with the encoded document embeddings.
        """
        df["headline_embeddings"] = self.get_embedding(df["headline_text"].tolist())
        self._save(df)
        return df
