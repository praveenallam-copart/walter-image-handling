import os
from logs import get_app_logger, get_error_logger
from dotenv import load_dotenv

load_dotenv()

class BaseClass:
    def __init__(self):
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.API_URL = os.getenv("API_URL")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")
        
        self.app_logger.info("Loggers Intialized and Access keys are retrieved...")