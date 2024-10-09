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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
from typing import List, Dict
from sentence_transformers import CrossEncoder
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

class PdfRetrieval(BaseClass):
    
    
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
                "Education" : [self.vector_store_chroma_Education, "Edu"],
                "Sports" : [self.vector_store_chroma_Sports, "Sports"],
                "Politics" : [self.vector_store_chroma_Politics, "politics"],
                "Environment" : [self.vector_store_chroma_Environment, "env"],
                "Others" : [self.vector_store_chroma_Others, "others"]
            }
            self.app_logger.info(f"Vector Databases are retrieved.....")  
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
    
    def reranker(self, retrieved_documents_content, retrieve_metadata, question):
        try:
            cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")
            pairs = [[question, document] for document in retrieved_documents_content]
            scores = cross_encoder.predict(pairs)
            self.app_logger.info("Reranking is done...")
            return scores
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 

    def get_information(self, retrieved_documents_content, retrieve_metadata, scores):
        try:
            scores_order = np.argsort(scores)[::-1]
            scores_order = scores_order[:3] # top 3
            self.information = []
            self.metadata = []
            for order in scores_order:
                self.information.append(retrieved_documents_content[order])
                self.metadata.append(retrieve_metadata[order])
            print(self.metadata)
            print(self.information)
            self.app_logger.info("got the information of top 3 relevant documents...")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 

    def retrieve_documents(self, question, category, name):
        try:
            vector_db = self.db_map[category][0]
            print(self.db_map[category][1])
            retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 50})   
            retrieved_documents = retriever.invoke(question)
            retrieved_documents_content = [str(document.page_content.replace("\n", " ")) for document in retrieved_documents]
            retrieve_metadata = [str(document.metadata) for document in retrieved_documents]
            scores = self.reranker(retrieved_documents_content, retrieve_metadata, question)
            self.get_information(retrieved_documents_content, retrieve_metadata, scores)
            self.app_logger.info(f"Relevant information retrieved....")
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        
    def rag(self, query, category, name):
        self.retrieve_documents(query, category, name)
        try:
            messages = [
                {
                    "role" : "system",
                    "content" : """
                    Using the following retrieved information, provide a concise and accurate response to the user's query. Focus on addressing the user's question directly, using the most relevant details from the provided documents. If multiple sources are relevant, synthesize the information while maintaining clarity. Avoid repeating information unless necessary for understanding. Ensure the response is coherent and fluent for the user.
                    """
                },
                {
                    "role" : "user",
                    "content" : f"""User Query: {query}

                            Retrieved Information:
                            1. {self.information[0]}
                            2. {self.information[1]}
                            3. {self.information[2]}"""
                    }
            ]
            response = self.llm.invoke(messages)
            content = response.content
            self.app_logger.info(f"LLM answered using relevant information...RAG!!")
            return response
        except Exception as e:
            self.error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
            raise 
        