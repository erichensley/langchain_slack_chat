import os
import logging
import configparser
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from utils.logging import setup_logging, log_message
from utils.gpt3_helpers import get_username, replace_user_ids_with_names
from utils.file_handler import get_messages_file_path, read_from_file
from utils.image_handler import trigger_modal, create_image, create_custom_images
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
def feed_message_to_openai(message, ack):
    ack()
    langchain_handler.handle_message(message, members)

@app.command("/image")
def make_image(ack, respond, command):
    ack()
    user_prompt = command["text"]
    channel_id = command["channel_id"]
    respond(text="Creating " + command["text"] + ", please wait...")
    image_url = langchain_handler.create_image(user_prompt)
    if image_url:
        langchain_handler.trigger_modal(channel_id, image_url, user_prompt)
    else:
        respond(text="Failed to create an image. Please try again.")

@app.command("/cimage")
def open_custom_image_modal(ack, body, client):
    ack()
    langchain_handler.trigger_modal(body["channel_id"], members)

@app.view("")
def handle_modal_submission(ack, body, client, logger):
    ack()
    langchain_handler.handle_modal_submission(body, client, logger)

if __name__ == "__main__":
    response = app.client.users_list()
    members = response["members"]
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
