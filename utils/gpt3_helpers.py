"""Helper functions for OpenAI and Replicate"""
import openai
import re
import logging
import tiktoken
import requests
import os
import configparser

from transformers import GPT2TokenizerFast
import replicate
from slack_sdk import WebClient
from utils.conversation_handler import load_history
from utils.file_handler import get_config_file_path, get_images_path, generate_image_url

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the INI file
config.read(get_config_file_path('api_keys.ini'))

os.environ["OPENAI_API_KEY"] = config.get('api_keys', 'OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = config.get('api_keys', 'GOOGLE_API_KEY')
os.environ["GOOGLE_CSE_ID"] = config.get('api_keys', 'GOOGLE_SEARCH_ID')
os.environ["SLACK_APP_TOKEN"] = config.get('api_keys', 'SLACK_APP_TOKEN')
os.environ["SLACK_BOT_TOKEN"] = config.get('api_keys', 'SLACK_BOT_TOKEN')
os.environ["REPLICATE_API_KEY"] = config.get('api_keys', 'REPLICATE_API_KEY')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"] )
rep = replicate.Client(api_token=os.environ["REPLICATE_API_KEY"])

# Return the number of tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Get word weights
def load_word_weights():
    word_weights = {}
    with open(get_config_file_path("word_weights.txt"), "r") as f:
        for line in f:
            token, weight = line.strip().split(",")
            weight = int(weight)

            word_weights[token] = weight
            word_weights[' ' + token] = weight
            word_weights[token.lower()] = weight
            word_weights[token.upper()] = weight

            # Add variations with punctuation marks
            punctuations = ['!', '.', ',', '?', ';', ':']
            for punc in punctuations:
                word_weights[token + punc] = weight
                word_weights[' ' + token + punc] = weight
                word_weights[token.lower() + punc] = weight
                word_weights[token.upper() + punc] = weight

    return word_weights

# Build weights
def build_logit_bias(tokenizer, word_weights):
    logit_bias = {}
    for word, weight in word_weights.items():
        token_id = tokenizer.encode(word)[0]
        logit_bias[token_id] = weight
    return logit_bias


# GPT-3 Embedding
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def generate_response_from_gpt3(message, username, previous_messages, prompt, max_tokens=4000):
    message_text = message['text']
    content = f"{username}: {message_text}"
    history = load_history()

    # Combine previous messages with the new message
    messages = prompt + previous_messages + history + [{"role": "user", "content": content}]
    #messages = prompt + [{"role": "user", "content": content}]
    # print("Prompt : " + str(prompt))
    # print("Previous Messages: " + str(previous_messages))
    # print("History: "+ str(history))
    conversation_history = "\n".join([msg["content"] for msg in messages])
    print("**** GPT-3 Prompt ****\n" + conversation_history)
    #Filter Words
    conversation_history = re.sub(r'(?i)AL!', '', conversation_history)
    # Calculate tokens
    tokens = len(tokenizer.encode(conversation_history))

    if tokens > max_tokens:
        tokens_to_remove = tokens - max_tokens

        # Iterate through messages in reverse order, removing tokens until the limit is reached
        while tokens_to_remove > 0:
            msg = history.pop()  # Remove the message from history instead of messages
            msg_tokens = len(tokenizer.encode(msg["content"]))
            tokens_to_remove -= msg_tokens

        # Rebuild messages and conversation_history after truncating history
        messages = prompt + previous_messages + history + [{"role": "user", "content": content}]
        conversation_history = "\n".join([msg["content"] for msg in messages])

        # Recalculate tokens for the new conversation_history
        tokens = len(tokenizer.encode(conversation_history))
    print(build_logit_bias(tokenizer, load_word_weights()))
    #Use Max tokens
    tokens_left = max_tokens - tokens
    # Use the OpenAI GPT-3 model to generate a response to the message
    response = openai.Completion.create(
        engine=openai_model_engine,
        prompt=conversation_history,
        max_tokens=tokens_left,
        presence_penalty=0.6,
        frequency_penalty=0.0,
        temperature=0.7,
        logit_bias = build_logit_bias(tokenizer, load_word_weights()),
        stop=None
    )

    # Get the generated response text
    response_text = response.choices[0].text.strip()

    return response_text

# Replace the Slack's user id their display name
def replace_user_ids_with_names(message, members):
    print("Replace User IDs Called")
    # Iterate through the list of users
    #print(str(members))
    for member in members:
        # Get the user's id
        user_id = member["id"]
        # Get the user's display name
        user_name = member["profile"]["display_name"] if member["profile"]["display_name"] else member["name"]
        # Replace the user's id with their display name
        message = re.sub(user_id, user_name, message)

    return message

def get_username(user_id, members):
    for member in members:
        if member["id"] == user_id:
            user_name = member["profile"]["display_name"] if member["profile"]["display_name"] else member["name"]
            return user_name
    return None