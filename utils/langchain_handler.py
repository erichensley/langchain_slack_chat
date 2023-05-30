import datetime
import os
import sys
import re
import copy
import traceback
import json
from typing import List, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.tools import tool
from utils.file_handler import load_json, read_from_file, get_config_file_path
from utils.gpt3_helpers import get_username, replace_user_ids_with_names
from utils.logging import setup_logging, log_message
from utils.image_handler import create_custom_images, create_image, trigger_image_modal

# Set up Prompt
prompt = read_from_file("config/prompt.txt").strip()

# Set up logging at the beginning of your script
logger = setup_logging()

@tool
def funny_response(input: str) -> str:
    """Returns a funny response."""
    return "Funny response"

search = WikipediaAPIWrapper()
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
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
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

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

class LangchainHandler:
    def __init__(self):
        self.models = load_json("config/models.json")
        self.user_dict = {}
        self.slack_members = {}
        self.last_user_parameters = {}

    def update_members(self, new_value):
        self.slack_members = new_value
 
    def handle_message(self, message, members):
        user = replace_user_ids_with_names(message["user"], members)
        user_id = str(message["user"])
        text = message["text"]
        response_text = agent_executor.run(input=text)
        log_message(user_id, user, text, response_text)
        return response_text

    def create_image(self, user_prompt):
        return create_image(user_prompt)

    def generate_block(self, model_name, parameter, details, values, last_prompt):
        print(f"Values for model '{model_name}': {values}")
        initial_value = ""
        block = {
            "type": "input",
            "block_id": f'{model_name}_{parameter}',
            "label": {"type": "plain_text", "text": parameter.capitalize()},
            "element": {},
            "hint": {"type": "plain_text", "text": details.get("description", "")}
        }
        if isinstance(values, dict) and f'{model_name}_{parameter}' in values:
            parameter_values = values[f'{model_name}_{parameter}']
            action_id = list(parameter_values.keys())[0]
            if details["type"] in ["text", "string", "integer", "number"]:
                initial_value = str(details["default"])
                if 'value' in parameter_values[action_id]:
                    initial_value = str(parameter_values[action_id]['value'])
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": initial_value},
                    "initial_value": initial_value
                }
            elif details["type"] == "dropdown":
                options = [{"text": {"type": "plain_text", "text": str(option)}, "value": str(option)} for option in details["options"]]
                initial_option = {"text": {"type": "plain_text", "text": str(details["default"])}, "value": str(details["default"])}
                if 'selected_option' in parameter_values[action_id]:
                    initial_option = parameter_values[action_id]['selected_option']
                block["element"] = {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select an option"},
                    "options": options,
                    "initial_option": initial_option
                }
            elif details["type"] == "prompt":
                initial_value = last_prompt or str(details["default"])
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": initial_value},
                    "initial_value": initial_value
                }
            else:
                # Default case for unknown types
                print(f"Unknown type '{details['type']}' for parameter '{parameter}' in model '{model_name}'. Defaulting to 'plain_text_input'.")
                initial_value = str(details.get("default", ""))
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": initial_value},
                    "initial_value": initial_value
            }
        else:
            print(f"No values found for parameter '{parameter}' in model '{model_name}'. Defaulting to 'plain_text_input'.")
            initial_value = str(details.get("default", ""))
            block["element"] = {
                "type": "plain_text_input",
                "placeholder": {"type": "plain_text", "text": initial_value},
                "initial_value": initial_value
        }
        #print("generate_block Last Prompt:", last_prompt)   
        #print(f"generate_block Generating block: {model_name}_{parameter}")
        #print("generate_block Initial Value:", initial_value)
        return block
    
    def open_custom_image_modal(self, ack, body, client):
        ack()
        # Store Channel ID and User ID in a dictionary
        self.user_dict[body['user_id']] = body['channel_id']
        user_id = body['user_id']
        last_parameters = load_last_used_values(user_id)
        last_prompt = last_parameters.get("prompt")

        selected_model = last_parameters.get("model_selection", {"JgSo": {"selected_option": {"value": "Kandinsky 2"}}})
        if isinstance(selected_model, dict):
            selected_model = selected_model[list(selected_model.keys())[0]]['selected_option']['value']

        blocks = self.generate_modal_blocks(selected_model, last_parameters, last_prompt)

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
    
    def generate_blocks_for_model(self, model_name, last_parameters, last_prompt):
        blocks = []
        parameters = self.models[model_name]
        for parameter, details in parameters.items():
            if parameter == "_model_id":
                continue  # skip the model ID
            block = self.generate_block(model_name, parameter, details, last_parameters, last_prompt)
            blocks.append(block)
        return blocks

    def handle_modal_submission(self, body, client, logger):
        try:
            members = self.slack_members
            user_id = body['user']['id']
            #Debug
            print ("handle_modal_submission user_id: " + user_id)
            username = get_username(user_id, members)
            values = body['view']['state']['values']
            logger.info(f"Values: {values}")
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
                    if self.models[model_name][parameter_name]["type"] == "integer":
                        parameter_value = int(parameter_value)
                    elif self.models[model_name][parameter_name]["type"] == "number":
                        parameter_value = float(parameter_value)
                    parameters[parameter_name] = parameter_value
            logger.info(f"Model: {model_name}, parameters: {parameters}")
            user_prompt = None
            for block_id, block_values in values.items():
                if block_id.endswith('_prompt'):
                    action_id = list(block_values.keys())[0]
                    user_prompt = block_values[action_id]['value']
                    break
            if user_prompt is None:
                logger.error("Failed to find prompt")
                return

            # Store the last used parameters and values
            save_last_used_values(user_id, values)
            logger.info(f"Last User Parameters: {values}")
            
            model_id = self.models[model_name]["_model_id"]
            channel_id = self.user_dict.get(body['user']['id'])
            client.chat_postEphemeral(channel=channel_id, user=user_id, text="Creating " + user_prompt + ", please wait...")
            urls = create_custom_images(model_id, parameters, self.models)
            message = ""
            for url in urls:
                message += f"{url}\n"
            if urls:  # Check if urls is not empty
                url = urls[0]  # Extract the first URL from the list
                trigger_image_modal(channel_id, url, f"{get_username(user_id, members)}: {user_prompt}")
            else:
                respond(text="Failed to create an image. Please try again.", client=client)
        except Exception as e:
            logger.error(f"Failed to handle modal submission: {e}")
            traceback.print_exc()
    def generate_modal_blocks(self, selected_model, last_parameters, last_prompt):
        model_options = [
            {
                "text": {
                    "type": "plain_text",
                    "text": model_name
                },
                "value": model_name
            }
            for model_name in self.models.keys()
        ]
        model_selection_block = {
                "dispatch_action": True,
                "type": "input",
                "block_id": "model_selection",
                "label": {
                    "type": "plain_text",
                    "text": "Model Selection"
                },
                "element": {
                    "type": "static_select",
                    "action_id": "model_selected", 
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Select a model"
                    },
                    "options": model_options,
                    "initial_option": model_options[0]  # select the first model by default
                }
            }

        # Generate blocks for the parameters of the selected model
        blocks = [model_selection_block] + self.generate_blocks_for_model(selected_model, last_parameters, last_prompt)

        return blocks

def save_last_used_values(user_id, last_used_values):
    config_file_path = get_config_file_path(f'{user_id}.json')
    with open(config_file_path, 'w') as f:
        json.dump(last_used_values, f)       

def load_last_used_values(user_id):
    try:
        config_file_path = get_config_file_path(f'{user_id}.json')
        with open(config_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # or some default values
