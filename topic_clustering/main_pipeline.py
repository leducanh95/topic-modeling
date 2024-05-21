from utils.contants import DATA_FILE
from document_retriever import DataRetriever


def mainipeline():
    retriever = DataRetriever(
        data_file=DATA_FILE,
        frac=0.1,
    )
    df = retriever.get_data()
    return df


if __name__ == "__main__":
    mainipeline()
