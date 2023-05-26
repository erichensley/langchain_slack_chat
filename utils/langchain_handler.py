import datetime
import os
import sys
import re
import traceback
from typing import List, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.tools import tool
from utils.file_handler import load_json, read_from_file
from utils.gpt3_helpers import get_username, replace_user_ids_with_names
from utils.logging import setup_logging, log_message

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
        self.user_channel_dict = {}

    def handle_message(self, message, members):
        user = replace_user_ids_with_names(message["user"], members)
        user_id = str(message["user"])
        text = message["text"]
        response_text = agent_executor.run(input=text)
        log_message(user_id, user, text, response_text)
        return response_text

    def create_image(self, user_prompt):
        return create_image(user_prompt)

    def trigger_modal(self, channel_id, members, image_url=None, prompt=None):
        if image_url:
            trigger_modal(channel_id, image_url, f"{get_username(user_id, members)}: {user_prompt}")
        else:
            trigger_modal(channel_id)

    def handle_modal_submission(self, body, client, logger):
        try:
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
                    if models[model_name][parameter_name]["type"] == "integer":
                        parameter_value = int(parameter_value)
                    elif models[model_name][parameter_name]["type"] == "number":
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
            model_id = models[model_name]["_model_id"]
            channel_id = self.user_channel_dict.get(body['user']['id'])
            client.chat_postEphemeral(channel=channel_id, user=user_id, text="Creating " + user_prompt + ", please wait...")
            urls = create_custom_images(model_id, parameters, models)
            message = ""
            for url in urls:
                message += f"{url}\n"
            if url:
                self.trigger_modal(channel_id, url, f"{get_username(user_id, members)}: {user_prompt}")
            else:
                respond(text="Failed to create an image. Please try again.")
        except Exception as e:
            logger.error(f"Failed to handle modal submission: {e}")
            traceback.print_exc()
