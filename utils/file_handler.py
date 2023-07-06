"""File Handler Functions"""
import json
import os
import random
import configparser
# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the INI file
config_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(config_dir, "..", "config")
config_file_path = os.path.join(folder, 'api_keys.ini')
config.read(config_file_path)

os.environ["IMAGE_HOST"] = config.get('api_keys', 'IMAGE_HOST')

def read_from_file(file_path):
    """
    Read the content of a file given its relative path.
    
    Args:
        file_path (str): The relative path to the file to be read.
        
    Returns:
        str: The content of the file as a string.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, "..", file_path)
    with open(abs_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def load_json(filepath):
    """
    Load JSON data from a file.
    
    Args:
        filepath (str): The path to the JSON file.
        
    Returns:
        dict: The JSON data as a dictionary.
    """
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def get_messages_file_path():
    """
    Get the absolute path to the messages log file.
    
    Returns:
        str: The absolute path to the messages log file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_messages_file = os.path.join(script_dir, "..", "log", "messages.txt")
    return abs_messages_file

def get_nexus_folder_path():
    """
    Get the absolute path to the 'nexus' folder.
    
    Returns:
        str: The absolute path to the 'nexus' folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_nexus_folder = os.path.join(script_dir, "..", "nexus")
    return abs_nexus_folder

def get_config_file_path(filename):
    """
    Get the absolute path to a configuration file.
    
    Args:
        filename (str): The name of the configuration file.
        
    Returns:
        str: The absolute path to the configuration file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_nexus_folder = os.path.join(script_dir, "..", "config")
    file_path = os.path.join(abs_nexus_folder, filename)
    return file_path

def get_images_path():
    """
    Get the absolute path to the 'images' folder.
    
    Returns:
        str: The absolute path to the 'images' folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(script_dir, "..", "images")
    return images_folder

def generate_image_url(file_name):
    """
    Generate the URL for an image hosted on a remote server.
    
    Args:
        file_name (str): The name of the image file.
        
    Returns:
        str: The URL of the hosted image.
    """
    url =  os.environ["IMAGE_HOST"] + file_name
    print(url)
    return url

def randomize_words(text):
    """
    Randomize the order of words in a given text.
    
    Args:
        text (str): The input text to randomize.
        
    Returns:
        str: The text with words randomized.
    """
    # Split the text into words
    words = text.split()

    # Shuffle the words
    random.shuffle(words)

    # Join the words back together into a single string
    randomized_text = ' '.join(words)

    return randomized_text

def print_step(step_number, step_name):
    print("\033[96mStep {}\033[0m".format(step_number))
    print("\033[93m{}\033[0m".format(step_name))
    print("\033[35m{}\033[0m".format('-' * 30))

def format_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}: {value}')

def print_color(text, color):
    color_dict = {
        'g': '\033[92m',  # Green
        'b': '\033[94m',  # Blue
        'c': '\033[96m',  # Cyan
        'm': '\033[95m',  # Magenta
        'y': '\033[93m',  # Yellow
        'r': '\033[91m',  # Red
        'w': '\033[0m',   # White
    }
    if color in color_dict:
        print(f"{color_dict[color]}{text}\033[0m")
    else:
        print("Invalid color. Please choose from g, b, c, m, y, r, or w.")
