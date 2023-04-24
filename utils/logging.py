import logging
import os

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
