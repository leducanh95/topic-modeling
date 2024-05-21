import pandas as pd
import os


class DataRetriever:
    def __init__(self, data_file, frac=None, min_date=None, max_date=None):
        self.data_file = data_file
        self.frac = frac
        self.min_date = min_date
        self.max_date = max_date

    def get_data(self):
        if not os.path.exists(self.data_file):
            raise Exception("File does not exist.")
        else:
            df = pd.read_csv(self.data_file)
            if self.frac is not None:
                df = df.sample(frac=0.1, random_state=42)
            df["publish_datetime"] = pd.to_datetime(df["publish_date"], format="%Y%m%d")
            df["publish_year"] = df["publish_datetime"].dt.year
            if self.min_date is not None:
                df = df[df["publish_datetime"] >= self.min_date]
            if self.max_date is not None:
                df = df[df["publish_datetime"] <= self.max_date]
        return df
