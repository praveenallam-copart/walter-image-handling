from fastapi import APIRouter, Depends
from tqdm.autonotebook import tqdm, trange
import fitz
import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import io
import re
from PIL import Image
from langchain_chroma import Chroma
import chromadb
from langchain.chains import LLMChain 
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
import uuid
import asyncio
from dependencies import Dependencies
from baseClass import BaseClass
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Document
import requests
from typing import List, Dict
# from sentence_transformers import CrossEncoder
import numpy as np
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticToolsParser
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from ImageHandling import ImageHandling

load_dotenv()

class PdfUploading(BaseClass):
    
    def __init__(self):
        super().__init__()
        self.llm = Dependencies().get_model(llm_service = "groq", llm = "groq/llama-3.1-70b-versatile")
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store_chroma_Education = Chroma(
                    collection_name="Education",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Sports = Chroma(
                    collection_name="Sports",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Politics = Chroma(
                    collection_name="Politics",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Environment = Chroma(
                    collection_name="Environment",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.vector_store_chroma_Others = Chroma(
                    collection_name="Others",
                    embedding_function=self.embeddings,
                    persist_directory="./walter-vector-storage",  # Where to save data locally, remove if not neccesary
                )
            self.db_map = {
                "Education" : self.vector_store_chroma_Education,
                "Sports" : self.vector_store_chroma_Sports,
                "Politics" : self.vector_store_chroma_Politics,
                "Environment" : self.vector_store_chroma_Environment,
                "Others" : self.vector_store_chroma_Others
            }
            self.app_logger.info(f"Vector Databases are retrieved.....")  
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    def encode_image(self, image_path):
        try:
            self.base64_image = base64.b64encode(image_path).decode("ascii")
            self.app_logger.info(f"base64 encoding, ascii decoding is done for given image")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def describe_image(self, base64_image):
        #"google/gemini-1.0-pro-002", "google/gemini-1.5-flash-001", google/gemini-1.5-pro-001"
        llm = Dependencies().get_model(llm_service = "groq", llm = "google/gemini-1.5-flash-001")
        try:
            messages = [
                {"role" : "system", "content" : "Your job is to extract all the information from the given image, including text with same structure. If you can't do it please do mention, don't make mistakes."},
                {"role" : "user", "content" : [
                    {"type": "text", "text" : "extract the information from the image (text and everything), with same structure present in image and give me a summary as well. Do not miss anything"},
                    {"type" : "image_url",
                    "image_url" : {
                        "url" : f"data:image/png;base64,{base64_image}",
                            },
                        }
                    ]
                }
            ]
            response = llm.invoke(messages)
            self.image_description = response.content
            self.app_logger.info(f"got the image description for above image_path....")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    def extract_images_text(self, filepath, output_file, name, user_id):
        try:
            total_summary = ""
            name = name.split(".")[0]
            self.app_logger.info(f"file {name=}...")
            output_folder = f"/export/home/saallam/image_handling/walter-image-handling/extracted_images/{name}" 
            document = fitz.open(filepath)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.app_logger.info(f"{output_folder=} created/ found...")
            self.combined_text = []
            for page_number in range(len(document)):
                self.app_logger.info(f"processing {page_number=}...")
                page = document.load_page(page_number)
                text = page.get_text()
                images = page.get_images(full = True)
                metadata = {"source" : f"{name}.pdf", "Page": page_number + 1, "Image": "No", "userid" : user_id, "summary" : "no"}
                page_info = Document(metadata = metadata, page_content = text)
                self.combined_text.append(page_info)
                total_summary += self.summary(text)
                if images:
                    self.app_logger.info(f"found images in {page_number=}...")
                    for image_index, image in enumerate(images):
                        self.app_logger.info(f"processing {image_index=}...")
                        xref = image[0]
                        base_image = document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_extension = base_image["ext"]
                        image_filename = f"{name}_page_{page_number+1}_image_{image_index+1}.{image_extension}"
                        image_filepath = os.path.join(output_folder, image_filename)
                        with open(image_filepath, "wb") as file:
                            file.write(image_bytes)
                        self.app_logger.info(f"image was written into {image_filepath=}...")
                        self.encode_image(image_bytes)
                        self.describe_image(self.base64_image)
                        metadata = {"source" : f"{name}", "Page": page_number + 1, "Image": image_filename, "userid" : user_id, "summary" : "no"}
                        total_summary += f"\n{self.summary(self.image_description)}\n"
                        page_info = Document(metadata = metadata, page_content = self.image_description)
                        self.combined_text.append(page_info)
            self.refined_summary = self.summary(total_summary)
            metadata = {"source" : f"{name}", "Page": None, "Image": "no", "userid" : user_id, "summary" : "yes"}
            page_info = Document(metadata = metadata, page_content = self.refined_summary)
            
            self.app_logger.info("Processing done ==> extract_images_text")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
    
    def pdf_loader(self, filepath, name, user_id):
        self.extract_images_text(filepath, "extracted_images", name, user_id)
        # loader = PyPDFLoader(filepath)
        # self.document = loader.load()
        # self.document.extend(self.combined_text)

    def text_splitter(self):
        # hyper parameter tuning for better chunk_size and chunk_overlap, custom text splitter 
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 700,
                chunk_overlap = 100,
                separators = ["\n\n", "\n"]
            )
            
            self.splits = text_splitter.split_documents(self.combined_text)
            self.app_logger.info("splitting done using recursive character text splitter...")
            self.app_logger.info(f"The lenght of documents (total number of pages) : {len(self.combined_text)}")
            self.app_logger.info(f"The lenght of splits (total number of pages after splits) : {len(self.splits)}")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise

    def load_vectorstore(self, vector_db):
        try:    
            uuids = [str(uuid4()) for _ in range(len(self.splits))]
            vector_db.add_documents(documents=self.splits, ids=uuids)
            self.app_logger.info(f"Splits are stored into the vector databse...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
            
    def store_data(self, downloadUrl, name, user_id):
        try:
            response = requests.get(downloadUrl)
            filepath = f"/export/home/saallam/image_handling/walter-image-handling/uploaded-pdfs/{name}.pdf"
            with open(filepath, 'wb') as pdf_file:
                pdf_file.write(response.content)
            self.app_logger.info(f"file {name=} is written...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
        self.pdf_loader(filepath, name, user_id)
        self.text_splitter()
        category = self.category(self.refined_summary)
        vector_db = self.db_map[category]
        self.load_vectorstore(vector_db = vector_db)
        self.app_logger.info(f"stored data successfully....")
        return self.refined_summary
        
    def summary(self, input_info):
        try:
            prompt = ChatPromptTemplate([
                ("system", """You are an advanced summarization expert trained to extract key insights without errors or hallucinations. Your task is to generate a concise and accurate summary of the provided input, ensuring that all critical information is retained. The summary should convey a clear understanding of the input’s content, reflecting all significant points and details.

                Follow these guidelines:

                Precision: Capture the essential ideas without omitting any relevant information.
                Clarity: Ensure that anyone reading the summary can easily comprehend the main points and overall message of the input.
                Accuracy: Only use the provided input; avoid adding any information that isn't explicitly present.
                Transparency: If you cannot produce a reliable summary based on the input, state that clearly instead of attempting to proceed."""),
                ("human", "input: {input}"),
            ])

            summary_llm = prompt | self.llm | StrOutputParser()
            response = summary_llm.invoke({"input" : input_info})
            self.app_logger.info(f"got the summary for the given input_info....")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
            
    
    def category(self, refined_summary):
        try:
            prompt = ChatPromptTemplate([
                ("system", """You are an expert text classifier with the ability to accurately categorize information based on its content. Your task is to analyze the given input and classify it into one of the following categories: Education, Sports, Politics, Environment, or Others.

                To guide your classification, refer to the descriptions and examples for each category below:

                Education: Content covering any educational topic, from foundational subjects like mathematics, languages, and sciences, to advanced fields like machine learning, probability, or other academic disciplines. This also includes materials related to educational systems, policies, and teaching methods.

                Examples:
                A tutorial on basic arithmetic or calculus.
                A textbook on machine learning algorithms.
                A study on language acquisition or grammar.
                Research on the probability theory or quantum physics.
                Discussions on school curriculums or higher education policies.
                Sports: Content focusing on athletic activities, sports competitions, fitness, or related events.

                Examples:
                A match report from a football or cricket tournament.
                An analysis of an athlete's performance or training routine.
                A review of major sporting events like the Olympics or the World Cup.
                Politics: Content involving governance, political ideologies, government policies, elections, or international relations.

                Examples:
                Discussions on legislative changes or political debates.
                An analysis of global political dynamics or elections.
                Commentary on foreign policies or political conflicts.
                Environment: Content that addresses issues related to ecology, climate change, conservation, or sustainability.

                Examples:
                Reports on climate change or its impacts on ecosystems.
                Articles on renewable energy, conservation efforts, or environmental policies.
                Discussions on global sustainability initiatives.
                Others: If the content does not align with any of the categories (Education, Sports, Politics, Environment), classify it under "Others."

                Examples:
                Content related to technology trends, entertainment, business, or lifestyle.
                Discussions on art, culture, or general non-political news.
                Your task: Review the input text carefully and classify it into one of these five categories. If the content does not clearly belong to Education, Sports, Politics, or Environment, assign it to "Others".
                The output should be in [Education, Sports, Politics, Environment, Others], do not give any extra information in the output.
                The output should be in string format eg: Education
                """),
                ("human", "input: {input}"),
            ])

            category_llm = prompt | self.llm | StrOutputParser()
            category = category_llm.invoke({"input" : refined_summary})
            self.app_logger.info(f"the given document is of {category=}")
            return category
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise
        
        