import json
from tqdm import tqdm

def load_and_filter_data(filename, start_key=None):
    """
    Load JSON data from a file and optionally filter it based on a starting key or index.

    Args:
        filename (str): The name of the JSON file.
        start_key (str, optional): The key to start iterating from. Defaults to None.

    Returns:
        dict: The filtered JSON data dictionary.

    The function opens the JSON file, loads its contents, and then filters the resulting dictionary
    based on the provided `start_key` . If no filtering is required, the entire
    dictionary is returned.

    The filtering is done by creating a new dictionary that only contains the key-value pairs with
    keys greater than or equal to `start_key`.
    """

    # Open the JSON file and load its contents
    with open(filename, 'r') as file:
        data = json.load(file)

    # Filter the data based on the provided start_key or start_index
    if start_key:
        # If a start_key is provided, filter the dictionary
        filtered_data = {key: value for key, value in data.items() if int(key) >= start_key}
    else:
        # No filtering is required, return the entire dictionary
        filtered_data = data

    return filtered_data



def iterate_file(filename, max_iter=None, start_key=None):
    """
    Opens the specified JSON file and iterates through each key-value pair.

    Args:
        filename (str): The name of the JSON file.
        max_iter (int, optional): The maximum number of iterations. If less than 0,
            it will be set to the total number of items in the JSON file. Defaults to None.
        start_key (str, optional): The key to start iterating from. Defaults to None.

    Yields:
        tuple: A tuple containing the key (ID) and the value ("Index-in-FFHQ").
    """

    # If max_iter is less than 0, set it to the total number of items in the JSON file
    if max_iter is None or max_iter < 0:
        with open(filename, 'r') as file:
            data = json.load(file)
            max_iter = len(data) if max_iter < 0 else max_iter
            print(f"Total items: {max_iter}")

    # Iterate through the filtered data and yield the key-value pairs
    with tqdm(total=max_iter, unit="pics") as progress_bar:
        filtered_data = load_and_filter_data(filename, start_key=start_key)
        for key, value in filtered_data.items():
            progress_bar.update(1)
            yield key, value['Index-in-FFHQ']

