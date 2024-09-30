import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from logs import get_app_logger, get_error_logger
from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage
from typing import Dict
import base64
import httpx
import asyncio
from dependencies import Dependencies
from baseClass import BaseClass
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

class CSVHandling(BaseClass):
    
    def __init__(self):
        super().__init__()
    
    async def readCsv(self, csvUrl):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(csvUrl)
                self.app_logger.info(f"csv response => {response.status_code}")
                with open("/export/home/saallam/image_handling/walter-image-handling/titanic.csv", "wb") as file:
                    file.write(response.content)
                self.app_logger.info(f"csv file written...")
            except Exception as e:
                self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                raise 
            
    async def csvAgent(self, query, csvUrl):
        await self.readCsv(csvUrl)
        llm = Dependencies().get_model(llm_service = "groq")
        try:
            agent = create_csv_agent(
                ChatOpenAI(temperature=0),
                "/export/home/saallam/image_handling/walter-image-handling/titanic.csv",
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True
            )
            self.app_logger.info(f"agent created...")
            response = agent.run(query)
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        