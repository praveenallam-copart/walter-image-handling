import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from logs import get_app_logger, get_error_logger
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage
from typing import Dict
import base64
import httpx
import asyncio

load_dotenv()

class Dependencies:
    
    def __init__(self):
        self.app_logger = get_app_logger()
        self.error_logger = get_error_logger()

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.API_URL = os.getenv("API_URL")
        self.GENIE_ACCESS_TOKEN = os.getenv("GENIE_ACCESS_TOKEN")

    def get_timestamp(self):
        return str(datetime.now())

    def get_model(self, llm_service:str, llm:str = None):
        #  Langchian LLM, Code Genie or OpenAI
        if llm is None or llm == "":
            llm = "gpt-4o-mini" if llm_service == "langchain" else "google/gemini-1.5-pro-001"
        try:
            client = ChatOpenAI(model=llm, base_url="http://copartcodegenapi-ws.c-qa4.svc.rnq.k8s.copart.com/v1", api_key = self.GENIE_ACCESS_TOKEN) \
                    if llm_service == "groq" else \
                        ChatOpenAI(model = llm, api_key = self.OPENAI_API_KEY, temperature = 0)
            self.app_logger.info(f"LLM initialised...{llm_service} @ {self.get_timestamp()}")
            return client
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
