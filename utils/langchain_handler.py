import datetime
import os
import sys
import re
import copy
import typing
import traceback
import json
from typing import List, Union, Dict, Any
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
    MODEL_SELECTION_BLOCK_ID = "model_selection" 
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

    def generate_block(self, model_name: str, parameter: str, details: Dict[str, Any], values: Dict[str, Any], last_prompt: str) -> Dict[str, Any]:
        """
        Function to generate block.
        :param model_name: The name of the model.
        :param parameter: The parameter.
        :param details: The details.
        :param values: The values.
        :param last_prompt: The last prompt.
        :return: The block.
        """
        print(f"Generating block for model: {model_name}, parameter: {parameter}")  # Debugging line
        print(f"Parameter details: {details}")  # Debugging line
        print(f"Values: {values}")  # Debugging line
        initial_value = ""
        block = {
            "type": "input",
            "block_id": f'{model_name}_{parameter}',
            "label": {"type": "plain_text", "text": parameter.capitalize()},
            "element": {},
            "hint": {"type": "plain_text", "text": details.get("description", "")}
        }
        # Check if model_name is in values and parameter is in values[model_name]
        if isinstance(values, dict) and parameter in values:
            # Get the value of the parameter
            parameter_value = values[parameter]
            print(f"Parameter value: {parameter_value}")  
            if details["type"] in ["text", "string", "integer", "number"]:
                initial_value = str(details["default"])
                initial_value = str(parameter_value)
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": initial_value},
                    "initial_value": initial_value
                }
            elif details["type"] == "dropdown":
                options = [{"text": {"type": "plain_text", "text": str(option)}, "value": str(option)} for option in details["options"]]
                initial_option = {"text": {"type": "plain_text", "text": str(parameter_value)}, "value": str(parameter_value)}
                block["element"] = {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select an option"},
                    "options": options,
                    "initial_option": initial_option
                }
            elif details["type"] == "prompt":
                initial_value = last_prompt or str(details["default"])
                initial_value = str(parameter_value)
                block["element"] = {
                    "type": "plain_text_input",
                    "placeholder": {"type": "plain_text", "text": initial_value},
                    "initial_value": initial_value
                }
            else:
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
        return block

    
    def open_custom_image_modal(self, ack, body, client):
        ack()
        # Store Channel ID and User ID in a dictionary
        self.user_dict[body['user_id']] = body['channel_id']
        user_id = body['user_id']

        # Load last used values for all models
        last_parameters_all_models = load_all_values(user_id)
        print(f"Retrieved last_parameters_all_models: {last_parameters_all_models}")

        # Get last used model
        selected_model = last_parameters_all_models.get('last_used_model', 'Kandinsky 2')
        print(f"Open Custom Image Selected model: {selected_model}")
        last_parameters = load_last_used_values(user_id, selected_model)
        print(f"Open Custom Image last_parameters: {last_parameters}")
        # Update the last parameters of the selected model with the current parameters
        #last_parameters = last_parameters_all_models.get(selected_model,
        #{}).get(selected_model, {})
        print(user_id)
        current_parameters = self.collect_parameters(last_parameters, selected_model, user_id)

        print(f"Retrieved current_parameters: {current_parameters}")
        last_prompt = current_parameters.get("prompt")
        print(f"Open_Custom_Image Selected model: {selected_model}")

        blocks = self.generate_modal_blocks(selected_model, current_parameters, last_prompt)

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
            user_id, values = extract_values_from_body(body)
            logger.info(f"Values: {values}")

            #model_selection_block_id = "model_selection"
            model_name = get_model_name(values, self.MODEL_SELECTION_BLOCK_ID)

            parameters = self.collect_parameters(values, model_name, user_id)
            logger.info(f"Model: {model_name}, parameters: {parameters}")
            print(f"Loaded parameters: {parameters}")
            user_prompt = find_user_prompt(values)
            if user_prompt is None:
                logger.error("Failed to find prompt")
                return

            # Store the last used parameters and values
            save_last_used_values(user_id, model_name, values)
            logger.info(f"Last User Parameters: {values}")

            # After updating last_parameters
            last_parameters = load_last_used_values(user_id, model_name)
            print(f"Updated last_parameters: {last_parameters}")
            
            model_id = self.models[model_name]["_model_id"]
            channel_id = self.user_dict.get(body['user']['id'])
            client.chat_postEphemeral(channel=channel_id, user=user_id, text=f"Creating {user_prompt}, please wait...")
            urls = create_custom_images(model_id, last_parameters, self.models)
            message = "\n".join(urls)
            
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
                    "initial_option": {
                        "text": {
                            "type": "plain_text",
                            "text": selected_model
                        },
                        "value": selected_model
                    }
                }
            }

        # Generate blocks for the parameters of the selected model
        blocks = [model_selection_block] + self.generate_blocks_for_model(selected_model, last_parameters, last_prompt)

        return blocks

    def get_value_for_parameter(self, user_id, model_name, parameter_name):
        last_used_values = load_last_used_values(user_id, model_name)
        parameter_block_id = f'{model_name}_{parameter_name}'
        if parameter_block_id in last_used_values:
            parameter_action_id = list(last_used_values[parameter_block_id].keys())[0]
            parameter_value = last_used_values[parameter_block_id][parameter_action_id].get('value')
            if parameter_value is None:
                parameter_value = last_used_values[parameter_block_id][parameter_action_id]['selected_option']['value']
        else:
            parameter_value = self.models[model_name][parameter_name]["default"]
            
        if self.models[model_name][parameter_name]["type"] == "integer":
            parameter_value = int(parameter_value)
        elif self.models[model_name][parameter_name]["type"] == "number":
            parameter_value = float(parameter_value)
        return parameter_value

    def collect_parameters(self, values: Dict[str, Any], model_name: str, user_id: str) -> Dict[str, Any]:
        parameters = {}
        for parameter_name, parameter_value in values.items():
            if parameter_name != 'last_used_model':
                parameters[parameter_name] = parameter_value
        print("collect_parameters called")
        print(parameters)
        return parameters


def save_last_used_values(user_id, model_name, values):
    # Load all previously saved values.
    user_values = load_all_values(user_id)

    # Prepare new values dictionary for the model
    new_values = {}
    for parameter, parameter_details in values.items():
        if parameter.startswith(model_name):
            # For each parameter, get the actual value
            for action_id, action_details in parameter_details.items():
                if 'value' in action_details:
                    # The parameter name is derived by removing the model_name_ prefix
                    parameter_name = parameter.replace(f"{model_name}_", "")
                    new_values[parameter_name] = action_details['value']

    # Update the values for this model.
    user_values[model_name] = new_values
    user_values['last_used_model'] = model_name

    # Save the updated values.
    with open(get_config_file_path(f'{user_id}.json'), 'w') as file:
        json.dump(user_values, file)

def load_last_used_values(user_id, model_name):
    last_parameters_all_models = load_all_values(user_id)
    model_parameters = last_parameters_all_models.get(model_name, {})
    return model_parameters

def load_all_values(user_id):
    print("load_all_values called")
    try:
        with open(get_config_file_path(f'{user_id}.json')) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def extract_values_from_body(body: Dict[str, Any]) -> Dict[str, Any]:
    user_id = body['user']['id']
    values = body['view']['state']['values']
    return user_id, values

def get_model_name(values: Dict[str, Any], model_selection_block_id: str) -> str:
    model_selection_action_id = list(values[model_selection_block_id].keys())[0]
    model_name = values[model_selection_block_id][model_selection_action_id]['selected_option']['value']
    return model_name

def find_user_prompt(values: Dict[str, Any]) -> str:
    user_prompt = None
    for block_id, block_values in values.items():
        if block_id.endswith('_prompt'):
            action_id = list(block_values.keys())[0]
            user_prompt = block_values[action_id]['value']
            break
    return user_prompt