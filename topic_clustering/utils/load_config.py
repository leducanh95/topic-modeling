import json
import os


def load_config(config_path: str) -> tuple:
    """
    Load the configuration file and extract the values of variables.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        tuple: A tuple containing the values of variable1, variable2, and variable3.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")
    else:
        with open(config_path) as f:
            config = json.load(f)
        # input_info = config.get("input_info")
        # output_info = config.get("output_info")
        # # Assign the values to variables
        # raw_data_path = input_info.get("raw_data_path", None)
        # frac = input_info.get("frac", None)
        # min_date = input_info.get("min_date", None)
        # max_date = input_info.get("max_date", None)
        # processed_data_path = output_info.get("processed_data_path", None)
        # clustered_data_path = output_info.get("clustered_data_path", None)
        # report_path = output_info.get("report_path", None)

    return config


if __name__ == "__main__":
    config = load_config("config/config.json")
    # print(config)
