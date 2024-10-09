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
import json
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from baseClass import BaseClass

load_dotenv()

class Dependencies(BaseClass):
    
    def __init__(self):
        super().__init__()

    def get_timestamp(self):
        return str(datetime.now())

    def get_model(self, llm_service:str, llm:str = None):
        #  Langchian LLM, Code Genie or OpenAI
        if llm is None or llm == "":
            llm = "gpt-4o-mini" if llm_service == "langchain" else "google/gemini-1.5-pro-001"
        try:
            client = ChatOpenAI(model=llm, base_url="http://copartcodegenapi-ws.c-qa4.svc.rnq.k8s.copart.com/v1", api_key = self.GENIE_ACCESS_TOKEN, temperature = 0) \
                    if llm_service == "groq" else \
                        ChatOpenAI(model = llm, api_key = self.OPENAI_API_KEY, temperature = 0)
            self.app_logger.info(f"LLM initialised...{llm_service} @ {self.get_timestamp()}")
            return client
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def get_chat_history(self):
        try:
            with open("/export/home/saallam/image_handling/walter-image-handling/chat_history.json", "r") as f:
                self.complete_chat_history = json.load(f)
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
            
    def write_chat_history(self, chat_history):
        try:
            with open("/export/home/saallam/image_handling/walter-image-handling/chat_history.json", "w") as f:
                f.write(json.dumps(chat_history, indent=4))
            self.app_logger.info(f"upadted chat history...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def write_file_map(self, file_map):
        try:
            with open("/export/home/saallam/image_handling/walter-image-handling/file_map.json", "w") as f:
                f.write(json.dumps(file_map))
            self.app_logger.info(f"updated file map...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def history(self):
        self.get_chat_history()
        history = []
        try:
            for chat in self.complete_chat_history:
                content = HumanMessage(content = chat["content"]) if chat["role"] == "user" else chat["content"]
                history.append(content)
            self.app_logger.info(f"chat history transformed and retrieved...")
            return history, self.complete_chat_history
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
            
    def get_file_map(self):
        try:
            with open("/export/home/saallam/image_handling/walter-image-handling/file_map.json", "r") as f:
                file_map = json.load(f)
            self.app_logger.info(f"file map loaded...")
            return file_map
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    def chat_comlpletion(self, query, history):
        llm = self.get_model(llm_service = "groq", llm = "groq/llama-3.1-70b-versatile")
        try:
            prompt = """
            You are an AI assistant that provides answers based on previous conversations and the user's current question. Consider the conversation history provided, and answer the current query in context.

            Conversation History: {history}
            Current Query: {query}
            """
            prompt_template = ChatPromptTemplate.from_template(prompt)
            message = prompt_template.format_messages(query = query, history = history)
            response = llm.invoke(message)
            self.app_logger.info(f"chat completion done....got a response!!")
            return response.content
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
