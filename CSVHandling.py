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
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent
from langchain.agents.agent_types import AgentType  
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from pathlib import Path

load_dotenv()

class CSVHandling(BaseClass):
    
    def __init__(self):
        super().__init__()
        self.SQL_USER = os.getenv("SQL_USER")
        self.SQL_PASSWORD = os.getenv("SQL_PASSWORD")
        self.SQL_HOST = os.getenv("SQL_HOST")
        self.SQL_PORT = os.getenv("SQL_PORT")
        self.SQL_DATABASE = os.getenv("SQL_DATABASE")
    
    async def read_store_csv(self, csvUrl,name, engine):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(csvUrl)
                self.app_logger.info(f"csv response => {response.status_code}")
                with open(f"/export/home/saallam/image_handling/walter-image-handling/csv_files/{name}.csv", "wb") as file:
                    file.write(response.content)
                self.app_logger.info(f"csv file written...")
                # df = pd.read_csv(f"/export/home/saallam/image_handling/walter-image-handling/csv_files/{name}.csv")
                # df.to_sql(f"{name}", self.engine, index=False)
                self.app_logger.info(f"csv stored...")
            except Exception as e:
                self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                raise 
    
    def csv_agent(self, llm, name = None):
        try:
            folder_path = Path.cwd() / "csv_files"
            csv_files = []
            refined_summary = None
            for filename in os.listdir(folder_path.as_posix()):
                csv_files.append((folder_path / filename).as_posix())
            if csv_files:
                csv_agent = create_csv_agent(llm, csv_files, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True)
                if name:
                    refined_summary = csv_agent.invoke(f"get the brief summary fo the file {name}")["output"]
                self.app_logger.info(f"csv agent created, and got the summary...")
                return csv_agent, refined_summary
            else:
                return None, None
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def get_sql_connection(self, llm):
        try:
            engine = create_engine(url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(self.SQL_USER, self.SQL_PASSWORD, self.SQL_HOST, self.SQL_PORT, self.SQL_DATABASE))
            db = SQLDatabase(engine=engine)
            sql_agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
            self.app_logger.info(f"Connection to the {self.SQL_HOST} for user {self.SQL_USER} created successfully and got the db!!!")
            self.app_logger.info(f"Database = {db.dialect} \ntables = {db.get_usable_table_names()}")
            self.app_logger.info(f"SQL agent got created!!!")
            return engine, db, sql_agent
        except Exception as e:
            print("Connection could not be made due to the following error: ", e)
    
    def query_agent(self, query, csv_agent):
        try:
            # sql_response = sql_agent.invoke({"input": query})
            csv_response = csv_agent.invoke(query)
            csv_agent_response = csv_response["output"]
            self.app_logger.info(f"There is an answer for the query!!!")
            # return {"sql_agent_response" : sql_response["output"], "csv_agent_response" : csv_agent_response}
            return {"csv_agent_response" : csv_agent_response}
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise        