import os

from topic_clustering.utils.logger import get_logger

logger = get_logger(name="folder")


def create_folder(folder_path: str) -> None:
    """
    Create a folder at the specified path if it doesn't already exist.

    Args:
        folder_path (str): The path of the folder to be created.
    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Folder '{folder_path}' created successfully.")
    else:
        logger.info(f"Folder '{folder_path}' already exists.")
