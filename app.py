from fastapi import FastAPI, HTTPException, Depends, status
from logs import get_app_logger, get_error_logger, setup_logging
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import Union, List, Dict
from dependencies import Dependencies


class ImageContents(BaseModel):
    image_content: List
    access_token: str

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
def image_description(imageContents: ImageContents):
    
    try:
        description = Dependencies().describe_image(imageContents.image_content, imageContents.access_token, llm_service = "groq")["image_description"]
        return {"image_description" : description}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/chat-completions")
def chat_completions(chatCompletion: ChatCompletion):
    try:
        text = chatCompletion.text
        image_description = chatCompletion.image_description
        llm_service = "groq"
        response = Dependencies().chat_comlpletions(text, image_description, llm_service)   
        return {"response" : {"status" : "200 OK", "message" : response}}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise 