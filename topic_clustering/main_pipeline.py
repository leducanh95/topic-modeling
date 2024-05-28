import pandas as pd
from config.constants import DATA_FILE
from document_embedder import DocumentEmbedder
from document_retriever import DataRetriever
from utils.logger import get_logger

logger = get_logger(name="main_pipeline")


def main_pipeline() -> pd.DataFrame:
    """
    This function represents the main pipeline for the topic clustering process.

    It retrieves data, performs document embedding, and returns the processed dataframe.

    Returns:
        pd.DataFrame: The processed dataframe with headline embeddings.
    """
    retriever = DataRetriever(
        data_file=DATA_FILE,
        frac=0.1,
    )
    df = retriever.get_data()

    embedder = DocumentEmbedder()
    df["headline_embeddings"] = embedder.get_embedding(df["headline_text"].tolist())
    print(df.head(5))
    return df


if __name__ == "__main__":
    main_pipeline()
