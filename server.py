"""Main Server for Slack + Langchain"""
import datetime
import logging
import os
import sys
import configparser
import re
from typing import List, Union
import traceback

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.tools import tool

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

#Load Custom Handlers
from utils.logging import setup_logging, log_message
from utils.gpt3_helpers import get_username, replace_user_ids_with_names
from utils.file_handler import get_messages_file_path, read_from_file
from utils.image_handler import trigger_modal, create_image, create_custom_images

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

# Set up Prompt
prompt = read_from_file("config/prompt.txt").strip()

# Define members as global variable
members = []

search = WikipediaAPIWrapper()

def funny_response(query: str) -> str:
    """Looks for any response that is supposed to make people laugh and is funny"""
    return f"Results for funny_response: {query}"

#search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="""useful for when you need to answer
          questions about current events or the current
            state of the world""",
    ),
    Tool(
        name="Funny Response",
        func=funny_response,
        description="""Looks for any response that is supposed to make people laugh and is funny""",
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")

# Create a custom prompt template class
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=prompt,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
    
llm = OpenAI(temperature=0)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
# Slack
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
app = App(token=os.environ["SLACK_BOT_TOKEN"])

user_channel_dict = {}
models = {
    "Kandinsky 2": {
        "_model_id": "cjwbw/kandinsky-2:65a15f6e3c538ee4adf5142411455308926714f7d3f5c940d9f7bc519e0e5c1a",
        "prompt": {
            "type": "text",
            "default": "red cat, 4k photo",
            "description": "Enter your creative prompt here. This is what the model will use to generate the image."
        },
        "num_inference_steps": {
            "type": "integer",
            "default": 50,
            "description": "The number of steps the model will take during the inference process."
        },
        "guidance_scale": {
            "type": "number",
            "default": 4,
            "description": "A parameter that determines the weight of the guidance from the input prompt."
        },
        "scheduler": {
            "type": "dropdown",
            "options": ["ddim_sampler", "p_sampler", "plms_sampler"],
            "default": "p_sampler",
            "description": "The scheduler determines the optimization method used during inference."
        },
        "prior_cf_scale": {
            "type": "integer",
            "default": 4,
            "description": "A parameter that influences the scale of the prior counterfactuals in the model."
        },
        "width": {
            "type": "dropdown",
            "options": [256, 288, 432, 512, 576, 768, 1024],
            "default": 512,
            "description": "The width of the output image in pixels."
        },
        "height": {
            "type": "dropdown",
            "options": [256, 288, 432, 512, 576, 768, 1024],
            "default": 512,
            "description": "The height of the output image in pixels."
        },
    },
    # add more models here
}

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
        respond(text="Creating " + command["text"] + ", please wait...")
        image_url = create_image(user_prompt)
        if image_url:
            # Trigger the modal after the image is ready
            trigger_modal(channel_id, image_url, f"{username}: {user_prompt}")
              # Pass the channel_id here
        else:
            respond(text="Failed to create an image. Please try again.")
    #Custom Image
    @app.command("/cimage")
    def open_custom_image_modal(ack, body, client):
        ack()
        #Store ChAanel ID and User ID in a dictionary
        user_channel_dict[body['user_id']] = body['channel_id']
        def generate_block(model_name, parameter, details):
            block = {
                "type": "input",
                "block_id": f'{model_name}_{parameter}',
                "label": {"type": "plain_text", "text": parameter.capitalize()},
                "element": {},
                "hint": {"type": "plain_text", "text": details.get("description", "")}
            }
            if details["type"] in ["text", "string", "integer", "number"]:
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": str(details["default"])},
                    "initial_value": str(details["default"])
                }
            elif details["type"] == "dropdown":
                block["element"] = {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select an option"},
                    "options": [{"text": {"type": "plain_text", "text": str(option)}, "value": str(option)} for option in details["options"]],
                    "initial_option": {"text": {"type": "plain_text", "text": str(details["default"])}, "value": str(details["default"])}
                }
            else:
                raise ValueError(f"Unknown detail type: {details['type']}")
            return block

        # Build the modal view based on the models
        blocks = [
            {
                "type": "input",
                "block_id": "model_selection",
                "label": {"type": "plain_text", "text": "Select model"},
                "element": {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select a model"},
                    "options": [{"text": {"type": "plain_text", "text": name}, "value": name} for name in models.keys()],
                    "initial_option": {"text": {"type": "plain_text", "text": "Kandinsky 2"}, "value": "Kandinsky 2"}  # Set default model
                }
            },
        ]
        for model_name, parameters in models.items():
            for parameter, details in parameters.items():
                if parameter == "_model_id":
                    continue  # skip the model ID
                block = generate_block(model_name, parameter, details)
                blocks.append(block)

        client.views_open(
            trigger_id=body["trigger_id"],
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

    @app.view("")
    def handle_modal_submission(ack, body, client, logger):
        ack()
        try:
            global models
            user_id = body['user']['id']
            username = get_username(user_id, members)
            values = body['view']['state']['values']
            logger.info(f"Values: {values}")
            if models is None:
                print("Warning: No models returned by get_models()")
            else:
                print(list(models.keys())[0])

            model_selection_block_id = "model_selection"
            model_selection_action_id = list(values[model_selection_block_id].keys())[0]
            model_name = values[model_selection_block_id][model_selection_action_id]['selected_option']['value']

            parameters = {}
            for parameter_block_id, parameter_details in values.items():
                if parameter_block_id != model_selection_block_id:
                    parameter_name = parameter_block_id.replace(model_name + '_', '')
                    parameter_action_id = list(parameter_details.keys())[0]
                    parameter_value = parameter_details[parameter_action_id].get('value')
                    if parameter_value is None:
                        parameter_value = parameter_details[parameter_action_id]['selected_option']['value']

                    # Convert parameter value to its appropriate type
                    if models[model_name][parameter_name]["type"] == "integer":
                        parameter_value = int(parameter_value)
                    elif models[model_name][parameter_name]["type"] == "number":
                        parameter_value = float(parameter_value)

                    parameters[parameter_name] = parameter_value

            logger.info(f"Model: {model_name}, parameters: {parameters}")

            user_prompt = None
            # Iterate over all blocks
            for block_id, block_values in values.items():
                # Check if block_id ends with '_prompt'
                if block_id.endswith('_prompt'):
                    action_id = list(block_values.keys())[0]
                    user_prompt = block_values[action_id]['value']
                    break

            # check if user_prompt was found
            if user_prompt is None:
                logger.error("Failed to find prompt")
                return

            model_id = models[model_name]["_model_id"]
            # Send the message to the channel where the command was executed
            channel_id = user_channel_dict.get(body['user']['id'])
            client.chat_postEphemeral(channel=channel_id, user=user_id, text="Creating " + user_prompt + ", please wait...")
            urls = create_custom_images(model_id, parameters, models)
            # Prepare a message with the generated image URLs
            message = ""
            for url in urls:
                message += f"{url}\n"

            if url:
                # Trigger the modal after the image is ready
                trigger_modal(channel_id, url, f"{username}: {user_prompt}")
                # Pass the channel_id here
            else:
                respond(text="Failed to create an image. Please try again.")
        except Exception as e:
            # Log the exception
            logger.error(f"Failed to handle modal submission: {e}")
            traceback.print_exc()


    @app.message("U04DG1YC0VC")
    #@app.message(".*") - Only Responed to mentions for now
    def feed_message_to_openai(message, say, ack):
        print("Feed message to OpenAI called")
        ack()
        user = replace_user_ids_with_names(message["user"], members)
        user_id = str(message["user"])
        text = message["text"]
        response_text = agent_executor.run(input=text)
        log_message(user_id, user, text, response_text)
        say(response_text)

    @app.message(".*")
    #@app.message(".*") - Only Responed to mentions for now
    def feed_message_to_openai(message, ack):
        ack()
        user = replace_user_ids_with_names(message["user"], members)
        user_id = str(message["user"])
        text = message["text"]
        response_text = None
        print("\033[94m" + user + ": " + text + "\033[0m")
        log_message(user_id, user, text, response_text)
        

    if __name__ == "__main__":
        response = app.client.users_list()
        members = response["members"]
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()

except Exception as e:
        print(f"An error occurred: {e}")

finally:
    input("Press Enter to close the script...")