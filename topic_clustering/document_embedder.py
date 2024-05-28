from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """A class for encoding documents into embeddings \
        using SentenceTransformer model."""

    def __init__(self):
        """Initializes the DocumentEmbedder class."""
        self.model = self.__init_model()

    def __init_model(self) -> SentenceTransformer:
        """
        Initializes the SentenceTransformer model.

        Returns:
            SentenceTransformer: The initialized SentenceTransformer model.
        """
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model

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
