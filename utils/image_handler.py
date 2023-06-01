import os
import replicate
import configparser
import requests
import uuid
import shutil
from typing import Dict
from slack_sdk import WebClient

from utils.file_handler import get_config_file_path, get_images_path, generate_image_url

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the INI file
config.read(get_config_file_path('api_keys.ini'))

os.environ["SLACK_APP_TOKEN"] = config.get('api_keys', 'SLACK_APP_TOKEN')
os.environ["SLACK_BOT_TOKEN"] = config.get('api_keys', 'SLACK_BOT_TOKEN')
os.environ["REPLICATE_API_KEY"] = config.get('api_keys', 'REPLICATE_API_KEY')

client = WebClient(token=os.environ["SLACK_BOT_TOKEN"] )
rep = replicate.Client(api_token=os.environ["REPLICATE_API_KEY"])



def trigger_image_modal(channel_id, image_url, title):  # Update the function parameters
    try:
        response = client.chat_postMessage(
            channel=channel_id,  # Use the channel_id here
            text="Here's your image:",
            blocks=[
                {
                    "type": "image",
                    "title": {"type": "plain_text", "text": title},
                    "image_url": image_url,
                    "alt_text": "Generated image:  " + title,
                }
            ],
        )
    except Exception as e:
        print(f"Error opening modal: {e}")

def create_image(prompt):
    """Run Replicate Model and return URL from config"""
    print("Image Prompt: " + prompt)
    model_id = ("cjwbw/kandinsky-2:"
                "65a15f6e3c538ee4adf5142411455308926714f7d3f5c940d9f7bc519e0e5c1a")
    print(rep.run(model_id, input={"prompt": prompt}))
    output = rep.run(model_id, input={"prompt": prompt})
    url = generate_image_url(download_and_save_image(output))
    return url

def create_custom_images(model_id: str, parameters: Dict[str, str], models):
    """Run a specific Replicate Model with custom parameters and return a URL from
    config"""
    print(f"parameters inside create_custom_images: {parameters}")
    print("Creating custom image...")
    # Prepare the input for the model
    input_parameters = {}

    # The prompt parameter is common for all models
    input_parameters["prompt"] = parameters.pop("prompt", "default prompt")
    # Add the rest of the parameters
    for parameter, value in parameters.items():
        input_parameters[parameter] = value

    print(f"Model ID: {model_id}, Input Parameters: {input_parameters}")

    # Run the model and get the output
    output = rep.run(model_id, input=input_parameters)
    urls = []
    url = generate_image_url(download_and_save_image(output))
    urls.append(url)
    return urls


def download_and_save_image(image_url):
    try:
        print("Downloading image...")
        
        # If image_url is a list, extract the first URL
        if isinstance(image_url, list):
            image_url = image_url[0]

        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        file_extension = os.path.splitext(image_url)[-1]
        new_file_name = f"{uuid.uuid4()}{file_extension}"
        save_path = os.path.join(get_images_path(), new_file_name)

        with open(save_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)

        return new_file_name
    except Exception as e:
        print(f"Failed to download and save image: {e}")
        return None
