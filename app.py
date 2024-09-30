from fastapi import FastAPI, HTTPException, Depends, status
from logs import get_app_logger, get_error_logger, setup_logging
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pydantic import BaseModel
from typing import Union, List, Dict
from ImageHandling import ImageHandling
from dependencies import Dependencies
from CSVHandling import CSVHandling


class ImageContents(BaseModel):
    image_content: Dict
    access_token: str

class CSVContents(BaseModel):
    csvUrl: str
    query: Union[str, None] = None

class ChatCompletion(BaseModel):
    text: Union[str, None] = None
    image_description: Union[str, None] =  None
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

setup_logging()
app_logger = get_app_logger()
error_logger = get_error_logger()

@app.get("/time")     
def time():
    return {"time" : Dependencies().get_timestamp()}

@app.post("/image_description")
async def image_description(imageContents: ImageContents):
    try:
        describe_image = await ImageHandling().describe_image(imageContents.image_content, imageContents.access_token, llm_service = "langchain")
        return {"image_description" : describe_image["image_description"]}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/csv_handling")
async def csv_handling(csvContents: CSVContents):
    try:
        response = await CSVHandling().csvAgent(csvContents.query, csvContents.csvUrl)
        return {"response" : response}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/chat-completions")
async def chat_completions(chatCompletion: ChatCompletion):
    try:
        text = chatCompletion.text
        image_description = chatCompletion.image_description
        llm_service = "langchain"
        response = await ImageHandling().chat_comlpletions(text, image_description, llm_service)   
        return {"response" : {"status" : "200 OK", "message" : response}}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise 
    
@app.get("/sys-version")
def sys_version():
    return {"response" : sys.version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host = "0.0.0.0", port = 8000, reload = True)