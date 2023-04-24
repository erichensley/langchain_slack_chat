"""Main Server for Slack + Langchain"""
import datetime
import logging
import os
import sys
import configparser
import re

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

#Load Custom Handlers
from utils.logging import setup_logging
from utils.gpt3_helpers import trigger_modal, get_username, create_image, replace_user_ids_with_names
from utils.file_handler import get_messages_file_path

#Load API Keys
config = configparser.ConfigParser()
config.read('config/api_keys.ini')

os.environ["OPENAI_API_KEY"] = config.get('api_keys', 'OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = config.get('api_keys', 'GOOGLE_API_KEY')
os.environ["GOOGLE_CSE_ID"] = config.get('api_keys', 'GOOGLE_SEARCH_ID')
os.environ["SLACK_APP_TOKEN"] = config.get('api_keys', 'SLACK_APP_TOKEN')
os.environ["SLACK_BOT_TOKEN"] = config.get('api_keys', 'SLACK_BOT_TOKEN')

# Set up logging at the beginning of your script
logger = setup_logging()

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="""useful for when you need to answer
          questions about current events or the current
            state of the world""",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Slack
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
app = App(token=os.environ["SLACK_BOT_TOKEN"])


try:
    @app.event("app_mention")
    def handle_app_mention_events(body, logger):
        """Create a logger object"""
        logger = logging.getLogger(__name__)
        with open(get_messages_file_path(), "a") as log_file:
            logger.info(str(body) + "\n")
            log_file.write(str(body) + "\n")

    # Make an image
    @app.command("/image")
    def make_image(ack, respond, command):
        ack()
        user_prompt = command["text"]
        channel_id = command["channel_id"]  # Get the channel ID from the command
        user_id = command["user_id"]
        username = get_username(user_id,members)
        image_url = create_image(user_prompt)
        
        if image_url:
            # Send an initial message to the user
            respond(text="Processing your image, please wait...")

            # Trigger the modal after the image is ready
            trigger_modal(channel_id, image_url, f"{username}: {user_prompt}")
              # Pass the channel_id here
        else:
            respond(text="Failed to create an image. Please try again.")

    @app.message(".*")
    def feed_message_to_openai(message, say, ack):
        print("Feed message to OpenAI called")
        logger = logging.getLogger(__name__)
        ack()
        payload = list()
        with open(get_messages_file_path(), "a") as log_file:
            user = replace_user_ids_with_names(message["user"], members)
            print(message["user"])
            user_id = str(message["user"])
            print(user_id)
            text = message["text"]
            response_text = agent_chain.run(input=text)
            log_file.write(user + ": " + text + "\n")
            log_file.write("AI: " + response_text + "\n")
            print("OpenAI Response: " + (str(response_text)))
        say(response_text)


    if __name__ == "__main__":
        response = app.client.users_list()
        members = response["members"]
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()

except Exception as e:
        print(f"An error occurred: {e}")

finally:
    input("Press Enter to close the script...")