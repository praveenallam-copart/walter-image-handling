# logs.py

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logging():
    log_directory = './logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(level=logging.INFO)

    # stream handler for stdout
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    # create the error logger
    error_handler = TimedRotatingFileHandler(filename='logs/error.log', when='midnight', backupCount=10)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)

    # create the app logger
    app_handler = TimedRotatingFileHandler(filename='logs/app.log', when='midnight', backupCount=10)
    app_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app_logger = logging.getLogger('app_logger')
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(app_handler)

    # Add app.log handler to the stdout logger
    logger.addHandler(app_handler)

def get_error_logger():
    return logging.getLogger('error_logger')

def get_app_logger():
    return logging.getLogger('app_logger')