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
from pdfUploading import PdfUploading
from pdfRetrieval import PdfRetrieval
from llmRouter import llm_router
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticToolsParser
from dependencies import Dependencies

class ImageContents(BaseModel):
    image_content: Dict
    access_token: str

class CSVContents(BaseModel):
    csvUrl: str
    name: str

class ChatCompletion(BaseModel):
    text: Union[str, None] = None
    image_description: Union[str, None] =  None

class UploadingFiles(BaseModel):
    downloadUrl: Union[str, None] =  None
    name: Union[str, None] = None
    message: Union[str, None] = None
    user_id: Union[str]

class InfoRetrievals(BaseModel):
    query: Union[str, None]
    name: Union[str]
    
class Completion(BaseModel):
    query: str
    name: Union[str, None] = None
    
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
history, chat_history = Dependencies().history()
file_map = Dependencies().get_file_map()
llm = Dependencies().get_model(llm_service = "groq", llm = "groq/llama-3.1-70b-versatile")
# engine, db, sql_agent = CSVHandling().get_sql_connection(llm)
csv_agent, refined_summary = CSVHandling().csv_agent(llm)

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
        name = csvContents.name.split(".")[0]
        await CSVHandling().read_store_csv(csvContents.csvUrl, name, engine = None)
        csv_agent, refined_summary = CSVHandling().csv_agent(llm, csvContents.name)
        history.extend([HumanMessage(content=f"Uploading a new file name = {csvContents.name}"), refined_summary])
        chat_history.extend([{"role" : "user", "content" : f"Uploading a new file name = {csvContents.name}"}, {"role" : "assistant", "content" : refined_summary}])
        return {"status" : 200, "response" : f"Read and stored the {name=}"}
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise

@app.post("/query_csv")
def query_csv(completion: Completion):
    try:
        query = f"{completion.query}\nuse {completion.name} for reference"
        response = CSVHandling().query_agent(query, csv_agent)
        history.extend([HumanMessage(content=query), response])
        chat_history.extend([{"role" : "user", "content" : query}, {"role" : "assistant", "content" : response}])
        return {"status" : 200, "response" : response}
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

@app.post("/UploadingFile")
def uploading_file(uploadingFile: UploadingFiles):
    # filepath = "/export/home/saallam/image_handling/walter-image-handling/pdfs/All you need is attention.pdf"
    name = uploadingFile.name.split(".")[0]
    refined_summary = PdfUploading().store_data(uploadingFile.downloadUrl, name, uploadingFile.user_id)
    category = PdfUploading().category(refined_summary)
    history.extend([HumanMessage(content=f"{uploadingFile.message}\n {uploadingFile.name}"), refined_summary])
    chat_history.extend([{"role" : "user", "content" : f"{"Uploading a new file"}\n {uploadingFile.name}"}, {"role" : "assistant", "content" : refined_summary}])
    # Dependencies().write_chat_history(chat_history)
    file_map[name] = category
    Dependencies().write_file_map(file_map)
    return {"response" : refined_summary, "category" : category}

@app.post("/pdf-retrieval")
def pdf_retrieval(infoRetrieval: InfoRetrievals):
    name = infoRetrieval.name.split(".")[0]
    category = file_map[name]
    response = PdfRetrieval().rag(infoRetrieval.query, category, name)
    history.extend([HumanMessage(content=infoRetrieval.query), response.content])
    chat_history.extend([{"role" : "user", "content" : infoRetrieval.query}, {"role" : "assistant", "content" : response.content}])
    # Dependencies().write_chat_history(chat_history)
    return {"response" : response, "Status" : 200}

@app.post("/llm-router")
def llm_router_completion(chatCompletion: ChatCompletion):
    response = llm_router(chatCompletion.text, history)
    return {"response" : response}

@app.get("/chat-history")
def get_chat_history():
    return {"response" : chat_history}

@app.post("/completions")
def completions(completion: Completion):
    response = Dependencies().chat_comlpletion(completion.query, history)
    history.extend([HumanMessage(content=completion.query), response])
    chat_history.extend([{"role" : "user", "content" : completion.query}, {"role" : "assistant", "content" : response}])
    # Dependencies().write_chat_history(chat_history)
    return {"response" : response, "status" : 200}

@app.get("/updateHistory")
def updateHistory():
    Dependencies().write_chat_history(chat_history)
    return {"response" : {"status" : 200, "chat_history" : chat_history}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host = "0.0.0.0", port = 8000, reload = True)