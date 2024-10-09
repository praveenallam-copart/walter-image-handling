from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser, PydanticToolsParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dependencies import Dependencies
from logs import get_app_logger, get_error_logger

load_dotenv()

def llm_router(input_query, chat_history):
    llm = Dependencies().get_model(llm_service = "groq", llm = "groq/llama-3.1-70b-versatile")
    try:
        app_logger = get_app_logger()
        error_logger = get_error_logger()

        filename_schema = ResponseSchema(name = "filename", description = "Name of the uploaded pdf file the user input is referring to, if there is no file name return -1")
        action_schema = ResponseSchema(name = "action", description = "if type is pdf then determine whether to use 'retrieve' or 'uploading' or whether it is 'unclear' or you are using your 'own knowledge' to answer else return -1")
        query_schema = ResponseSchema(name = "query", description = "it is the input given by the user")
        type_schema = ResponseSchema(name = "type", description = "it is the type of the file pdf or csv or text")

        response_schema = [filename_schema, action_schema, query_schema, type_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        
        
        prompt_template = """
            You are an advanced language model designed to assist users with analyzing and responding to queries about various types of content, including PDFs, CSVs, and other documents. Users may upload or link files for you to reference. Based on the given query and chat_history find out whether the user is asking questions regarding already uploaded files (use the summary present to determine), if you think the query is not referring to any of the files uploaded from chat_history use your own knowledge to answer. If you are not sure ask user if he's referring to files uploaded and filename.

            if the filenmae is of type pdf follow below instructions

                    Key guidelines for handling queries realted to pdf files:

                    File Handling:
                    If a user uploads a new file, "Uploading a new file along with filename" should be stored in chat_history. So you can get filenames from that. example: Uploading a new file abc.pdf, Uploading a new file xyz.pdf etc
                    Contextual Answers for PDFs:
                    Use the summary: When a user asks general questions about a previously uploaded PDF (e.g., "explain," "what is the PDF about", "topics", "context"), retrieve the file's stored summary from the chat history to provide a response.
                    Specific Queries: For questions requiring specific details from a PDF, respond with "retrieve" to indicate that the system will fetch the relevant sections from the vector database.
                    Handling Ambiguity:
                    If it's unclear whether the user's query relates to an uploaded file, respond with "unclear" to indicate uncertainty about the file connection.
                    Knowledge-Based Answers:
                    If the query is unrelated to any file or doesn't require file retrieval, respond using your general knowledge. If a question has been asked before, use chat history to respond accurately without accessing the vector database.
                    Summary Usage Priority:
                    Prioritize using the stored file summary for general document-related questions to avoid unnecessary retrieval from the vector database, ensuring more accurate and relevant responses.
                    Your goal is to provide accurate, context-aware answers, leveraging file summaries for general questions and the vector database for specific ones. Use chat history to maintain context and improve responses.
            
            if the filenmae is of type csv follow below instructions

                    Key guidelines for handling queries realted to csv files:

                    File Handling:
                    If a user uploads a new file, "Uploading a new file along with filename" should be stored in chat_history. So you can get filenames from that. example: Uploading a new file abc.csv, Uploading a new file xyz.csv etc
                    Specific Queries: For questions requiring specific details from a PDF, respond with "retrieve" to indicate that the system will fetch the relevant sections from the vector database.
                    The column names can be found in history, so it makes clear to decide
                    Handling Ambiguity:
                    If it's unclear whether the user's query relates to an uploaded file, respond with "unclear" to indicate uncertainty about the file connection.
                    Knowledge-Based Answers:
                    If the query is unrelated to any file or doesn't require file retrieval, respond using your general knowledge. If a question has been asked before, use chat history to respond accurately without accessing the vector database.
                    Summary Usage Priority:
                    Prioritize using the stored file summary for general document-related questions to avoid unnecessary retrieval from the vector database, ensuring more accurate and relevant responses.
                    Your goal is to provide accurate, context-aware answers, leveraging file summaries for general questions and the vector database for specific ones. Use chat history to maintain context and improve responses.

            Just follow the instructions don't give additional information in the output

            query: {query}
            chat_history: {chat_history}
            instructions: {format_instructions}
        """
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        message = prompt_template.format_messages(query = input_query, chat_history = chat_history, format_instructions = format_instructions)

        # final_llm = prompt | llm | StrOutputParser()
        response = llm.invoke(message)
        response = output_parser.parse(response.content)
        app_logger.info(f"route for the given query is found....")
        return response
    except Exception as e:
        error_logger.error(f"An unexpected e occurred: {type(e).__name__, str(e)}")
        raise 
    
    
    # store chat_history, pdf_type in files
    # perfect pipeline for pdf handling
    # image handling testing
    # evalutaion, experimentation of different splitters