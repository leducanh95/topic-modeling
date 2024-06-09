import os

import pandas as pd

from topic_clustering.utils.logger import get_logger

logger = get_logger(name="document_retriever")


class DataRetriever:
    """This class retrieves and preprocesses data from a specified file."""

    def __init__(self, config: dict):
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
        input_info = config.get("input_info")
        input_dir = input_info.get("input_dir", None)
        self.data_file = os.path.join(
            input_dir, input_info.get("raw_data_filename", None)
        )
        self.frac = input_info.get("frac", None)
        self.min_date = input_info.get("min_date", None)
        self.max_date = input_info.get("max_date", None)

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
            logger.info(f"Data shape: {df.shape}")
        return df
