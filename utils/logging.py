import logging
import os
from utils.file_handler import get_messages_file_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set up logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_dir = os.path.join(SCRIPT_DIR, "..", "log")
    os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist

    log = os.path.join(log_dir, "app_log.txt")
    # File handler to log messages to a file
    file_handler = logging.FileHandler(log)
    file_handler.setFormatter(formatter)

    # Stream handler to log messages to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def log_message(user_id: str, user: str, message_text: str, response_text: str):
    logger = logging.getLogger(__name__)
    with open(get_messages_file_path(), "a") as log_file:
        log_file.write(f"{user}: {message_text}\n")
        if response_text is not None:
            log_file.write(f"AI: {response_text}\n")
    logger.info(f"User ID: {user_id}")
    if response_text is not None:
        logger.info(f"OpenAI Response: {response_text}")
