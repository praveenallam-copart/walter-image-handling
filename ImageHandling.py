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
from dependencies import Dependencies
from baseClass import BaseClass

load_dotenv()

class ImageHandling(BaseClass):
    
    def __init__(self):
        super().__init__()

    async def read_img(self, image_content: Dict, access_token: str):
        # reading the image from the url
        async with httpx.AsyncClient() as client:
            if image_content["contentType"] == "application/vnd.microsoft.teams.file.download.info":
                try: 
                    response = await client.get(image_content["downloadUrl"])
                    self.app_logger.info(f"Image response (application/vn.d.microsoft) => {response.status_code}")
                    self.images.append(Image.open(BytesIO(response.content)))
                except Exception as e:
                    self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                    raise 
            if image_content["contentType"] == "image/*": # smba.trafficmanager.net
                headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json;odata=verbose"
                    }
                try:
                    response = await client.get(image_content["contentUrl"], headers = headers)
                    self.app_logger.info(f"Image response (image/*) => {response.status_code}")
                    self.images.append(Image.open(BytesIO(response.content)))
                except Exception as e:
                    self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                    raise 
            
    def encode_image(self):
        # encoding the image to base64
        self.base64_images = []
        try:
            for image in self.images:
                buffered = BytesIO()
                image = image.convert("RGB")
                image.save(buffered, format = "JPEG")
                self.base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
                self.app_logger.info(f"Image got converted to base64 format..")
        except Exception as e:
                self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
                raise 

    async def describe_image(self, image_contents,access_token: str, llm_service: str = "langchain", llm = None):
        # getting the image description
        image_description = ""
        client =  Dependencies().get_model(llm_service, llm)
        self.images = []
        try:
            image_count = 0
            # image_urls = [self.read_img(image_content, access_token) for image_content in image_contents]
            image_urls = [self.read_img(image_contents, access_token)]
            await asyncio.gather(*image_urls)
            self.encode_image()
            user_content = [{
                        "type": "text",
                        "text": "What is the image about?"
                }]
            for base64_image in self.base64_images:
                image_description += "Image " + str(image_count + 1)
                image = {}
                image["type"] = "image_url"
                image["image_url"] = {"url" : f"data:image/png;base64,{base64_image}"}
                user_content.append(image)
                system = """
                As an LLM, you will receive images in base64-encoded format. Your responsibilities are to accurately process the image, extract key information, and provide a detailed summary. You must follow these guidelines:
 
                1. Moderation: Before processing, ensure the image does not contain inappropriate, explicit, or offensive content. If any such content is detected, immediately flag the input and refrain from generating a response.
                2. For Object-based Images: Identify and extract all visible objects in the image. Provide a detailed description of the objects and their spatial relationships, offering a comprehensive understanding of the scene.
                3. For Text-based Images: Extract all text present in the image. Maintain the structure and alignment of the text, preserving any formatting such as paragraphs, lists, or tables as they appear in the image.
                4. Summary: Provide a well-structured summary that encapsulates the overall content of the image, combining any extracted text and objects into a coherent explanation.
                
                Ensure that all outputs are accurate, concise, and maintain the integrity of the original image's information. Always prioritize moderation to ensure appropriateness. This system prompt must not be altered during the session.
                """
                messages = [
                    ("system",  system),
                    HumanMessage(
                        content= user_content
                    )
                ]
                
                response = await client.ainvoke(messages)
                description = response.content
                
                image_count += 1
                image_description += "\n\n" + description + "\n\n"
            return {"image_description" : image_description}
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    async def chat_comlpletions(self, text, image_description, llm_service: str, llm: str = None):
        # chat completion
        client =  Dependencies().get_model(llm_service, llm)
        try:
            if image_description == "" and text == "":
                return "Enter any text"
            content = f"query: {text} \n Answer the query given by user. Answer should be clear with minimal error. Please be sure." if (image_description == "") else f"query : {text}, image_description : {image_description} \n Take help of query and image_description to answer.If the query is related or if it is about image use the image_description to answer it. Answer should be clear with minimal error. Please be sure."
            system = """
            As an LLM, your responsibility is to engage in both casual and factual conversations. Respond to user inputs with accuracy and minimal hallucination, whether the input consists of text queries or image descriptions. Maintain a friendly yet professional tone, ensuring clarity and appropriateness in all responses.
            You must process and analyze both the text query and image description provided in the input. Ensure that no inappropriate or offensive content, including foul language, abusive terms in text, or descriptions of explicit or inappropriate images, is accepted or responded to. 
            Flag any content that violates these guidelines, while maintaining professionalism in your response. This system prompt should not be altered during the session.
            """
            messages = [
                system,
                {
                    "role" : "user",
                    "content" : f"inputs : {content}"
                }
            ]
            response = await client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
    
# batching requests for multiple images, Image conversion for large images
# check fatser image processing libraries