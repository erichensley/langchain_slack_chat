import os
import time
import logging
import configparser
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from utils.logging import setup_logging, log_message
from utils.gpt3_helpers import get_username, replace_user_ids_with_names
from utils.file_handler import get_messages_file_path, read_from_file
from utils.image_handler import trigger_image_modal, create_image, create_custom_images
from utils.langchain_handler import LangchainHandler

# Load API keys
config = configparser.ConfigParser()
config.read('config/api_keys.ini')

os.environ["OPENAI_API_KEY"] = config.get('api_keys', 'OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = config.get('api_keys', 'GOOGLE_API_KEY')
os.environ["GOOGLE_CSE_ID"] = config.get('api_keys', 'GOOGLE_SEARCH_ID')
os.environ["SLACK_APP_TOKEN"] = config.get('api_keys', 'SLACK_APP_TOKEN')
os.environ["SLACK_BOT_TOKEN"] = config.get('api_keys', 'SLACK_BOT_TOKEN')

# Set up logging at the beginning of your script
logger = setup_logging()

# Set up Slack
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize LangchainHandler
langchain_handler = LangchainHandler()

# Slack event handlers
@app.event("app_mention")
def handle_app_mention_events(body, logger):
    logger = logging.getLogger(__name__)
    with open(get_messages_file_path(), "a") as log_file:
        logger.info(str(body) + "\n")
        log_file.write(str(body) + "\n")

@app.message("U04DG1YC0VC")
def feed_message_to_openai(message, say, ack):
    ack()
    response_text = langchain_handler.handle_message(message, members)
    say(response_text)

@app.message(".*")
def handle_message_events(message, say, ack):
    ack()
    print(message)
    # Check if the message is a thread reply
    if "thread_ts" in message:
        # Check if the message starts with the keyword "change"
        if message["text"].startswith("change:"):
            # Extract the image URL and the prompt from the message
            image_url = message["attachments"][0]["image_url"]
            prompt = message["text"].split(":", 1)[1].strip()
            # Call the new function in LangchainHandler to handle the image modification request
            result = langchain_handler.handle_image_modification_request(image_url, prompt)
            # Post the result in the thread
            say(text=result, thread_ts=message["thread_ts"])
    else:
        # Handle non-threaded messages here
        pass

@app.command("/image")
def make_image(ack, respond, command):
    ack()
    user_prompt = command["text"]
    channel_id = command["channel_id"]
    user_id = command["user_id"]
    username = get_username(user_id,members)
    respond(text="Creating " + command["text"] + ", please wait...")
    title = username + ": " + command["text"]
    start_time = time.time()
    image_url = langchain_handler.create_image(user_prompt)
    generation_time = time.time() - start_time
    parameters = {"prompt": user_prompt}  # Add other parameters as needed
    if image_url:
        trigger_image_modal(channel_id, image_url, title, parameters, title)
    else:
        respond(text="Failed to create an image. Please try again.")

@app.command("/cimage")
def open_custom_image_modal(ack, body, client):
    ack()
    #langchain_handler.trigger_modal(body["channel_id"], members)'')
    langchain_handler.step1_open_custom_image_modal(ack, body, client)

@app.command("/punchout")
def make_punchout_image(ack, respond, command):
    ack()
    user_prompt = command["text"]
    channel_id = command["channel_id"]
    user_id = command["user_id"]
    username = get_username(user_id,members)
    respond(text="Creating " + command["text"] + ", please wait...")
    title = username + ": " + command["text"]
    start_time = time.time()
    image_url = langchain_handler.create_punchout_image(user_prompt)
    generation_time = time.time() - start_time
    parameters = {"prompt": user_prompt}  # Add other parameters as needed
    if image_url:
        trigger_image_modal(channel_id, image_url, title, parameters, title)
    else:
        respond(text="Failed to create an image. Please try again.")

@app.view("")
def step3_handle_submission(ack, body, client, logger):
    ack()
    langchain_handler.step3_handle_submission(body, client, logger)

@app.action("model_selected")
def handle_model_selection(ack, body, client):
    ack()
    user_id = body['user']['id']
    last_parameters = body['view']['state']['values']
    last_prompt = last_parameters.get("prompt")
    selected_model = last_parameters['model_selection']['model_selected']['selected_option']['value']
    print(f"model_selected Selected model: {selected_model}")
    print(f"model_selected Last parameters: {last_parameters}")
    blocks = langchain_handler.step2_generate_modal_blocks(selected_model, last_prompt, user_id)
    print(f"model_selected Blocks: {blocks}")
    client.views_update(
        view_id=body['view']['id'],
        view={
            "type": "modal",
            "title": {
                "type": "plain_text",
                "text": "Custom Image",
                "emoji": True
            },
            "submit": {
                "type": "plain_text",
                "text": "Submit",
                "emoji": True
            },
            "blocks": blocks
        }
    )


if __name__ == "__main__":
    response = app.client.users_list()
    members = response["members"]
    with open("members.json", "w") as file:
        json.dump(members, file)
    langchain_handler.update_members(members)
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
