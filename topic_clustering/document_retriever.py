import os

import pandas as pd
from utils.logger import get_logger

logger = get_logger(name="document_retriever")


class DataRetriever:
    """This class retrieves and preprocesses data from a specified file."""

    def __init__(
        self,
        data_file: str,
        frac: float = None,
        min_date: str = None,
        max_date: str = None,
    ):
        """
        Initialize the DataRetriever class.

        Args:
            data_file (str): The path to the data file.
            frac (float, optional): The fraction of data to sample. \
                Defaults to None.
            min_date (str, optional): The minimum date to filter the data. \
                Defaults to None.
            max_date (str, optional): The maximum date to filter the data. \
                Defaults to None.
        """
        self.data_file = data_file
        self.frac = frac
        self.min_date = min_date
        self.max_date = max_date

    def get_data(self) -> pd.DataFrame:
        """
        Load and preprocess the data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        logger.info(f"Loading data from {self.data_file}")
        if not os.path.exists(self.data_file):
            raise Exception("File does not exist.")
        else:
            df = pd.read_csv(self.data_file)
            if self.frac is not None:
                df = df.sample(frac=self.frac, random_state=42)
            df["publish_datetime"] = pd.to_datetime(df["publish_date"], format="%Y%m%d")
            df["publish_year"] = df["publish_datetime"].dt.year
            if self.min_date is not None:
                df = df[df["publish_datetime"] >= self.min_date]
            if self.max_date is not None:
                df = df[df["publish_datetime"] <= self.max_date]
            logger.info(
                f"Data loaded successfully. From {self.min_date} to {self.max_date} and sampled by {self.frac}"  # noqa E501
            )
        return df
