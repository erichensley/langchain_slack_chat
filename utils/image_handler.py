import os
import time
import traceback
import configparser
import requests
import uuid
import shutil
from typing import Dict
import replicate
from slack_sdk import WebClient

from utils.file_handler import get_config_file_path, get_images_path, generate_image_url, print_color

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the INI file
config.read(get_config_file_path('api_keys.ini'))

os.environ["SLACK_APP_TOKEN"] = config.get('api_keys', 'SLACK_APP_TOKEN')
os.environ["SLACK_BOT_TOKEN"] = config.get('api_keys', 'SLACK_BOT_TOKEN')
os.environ["REPLICATE_API_TOKEN"] = config.get('api_keys', 'REPLICATE_API_KEY')

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
    start_time = time.time()

    try:
        print_color("Image Prompt: " + prompt, "b")
        model_image = replicate.models.get("stability-ai/sdxl").versions.get("2f779eb9b23b34fe171f8eaa021b8261566f0d2c10cd2674063e7dbcd351509e")
        # Measure time taken by rep.run()
        start_rep_run = time.time()
        image = model_image.predict(prompt=prompt, scheduler='KarrasDPM', refine='expert_ensemble_refiner')
        end_rep_run = time.time()
        print_color(f"Time taken by Image Creation: {end_rep_run - start_rep_run} seconds", "y")
        # Measure time taken by download_and_save_image() and generate_image_url()
        start_image_processing = time.time()
        url = generate_image_url(download_and_save_image(image))
        end_image_processing = time.time()
        print_color(f"Time taken by download_and_save_image() and generate_image_url(): {end_image_processing - start_image_processing} seconds", "y")

        print_color(f"Total time taken by create_image(): {time.time() - start_time} seconds", "y")

        return url
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        print(traceback.format_exc())

def create_custom_images(model_id: str, parameters: Dict[str, Dict[str, str]]):
    """Run a specific Replicate Model with custom parameters and return a URL from
    config"""
    #print(f"parameters inside create_custom_images: {parameters}")
    #print("Creating custom image...")
    # Prepare the input for the model
    input_parameters = {}
    schema = {'prompt': 'string', 'num_inference_steps': 'integer', 'guidance_scale': 'number', 'prior_cf_scale': 'integer', 'scheduler': 'string', 'refine': 'string', 'high_noise_frac': 'number'}
    # Get the first (and only) key-value pair in the dictionary
    _, model_parameters = next(iter(parameters.items()))

    # Copy the parameters for this model and convert types based on schema
    for key, value in model_parameters.items():
        try:
            if schema[key] == 'integer':
                input_parameters[key] = int(value)
            elif schema[key] == 'number':
                input_parameters[key] = float(value)
            else:
                input_parameters[key] = value
        except KeyError:
            raise ValueError(f"Unexpected parameter '{key}' encountered. Please check your model parameters.")

    # New print statement for desired output
    print_color(f"Custom image created: Model '{model_id}'\nPrompt '{input_parameters.get('prompt')}'", "b")

    # Run the model and get the output
    output = rep.run(model_id, input=input_parameters)
    urls = []
    url = generate_image_url(download_and_save_image(output))
    urls.append(url)
    return {'urls': urls, 'parameters': input_parameters}

def download_and_save_image(image_url):
    try:
        print_color("Downloading image...", "g")
        
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
