from root import CONFIG_FILE_PATH
from topic_clustering.main_pipeline import main_pipeline


def main(config_file: str) -> None:
    """
    Main function to run the topic modeling pipeline.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        None
    """
    main_pipeline(config_file)


if __name__ == "__main__":
    main(config_file=CONFIG_FILE_PATH)
