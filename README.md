![app](https://github.com/user-attachments/assets/891b9244-76c9-4457-8f85-201d0d27840b)

### PDF Handling
If a user requests information regarding a PDF file by attaching the file and a query, the process would first involve uploading the PDF to the vector store and then answering the query. If the user is querying against a previously uploaded file, the system will provide an answer using the relevant PDF that was uploaded earlier

#### PDF Uploading:
![pdfUploading](https://github.com/user-attachments/assets/de0b72b5-001e-4317-a276-a41fd117959f)
* First, the PDF file is downloaded and stored. The system then iterates through each page of the PDF, extracting text from each one. If any images are present on a page, they are downloaded and stored with unique names. These images are then used to generate image descriptions. A summary is extracted from each page, and these are combined to create a complete summary of the document
* The combined summary is used to determine the category of the PDF, such as Education, Politics, or Others.
* The page content (text) and image descriptions are stored in Langchain's Document format, along with metadata
```
Document(metadata = {'Page': 1, 'source': 'Football Analysis.pdf', 'summary': 'no', 'userid': '743ae8a1-2864-4e54-950e-9a20275e9856'}, page_content = 'Dutch players are not as close....')
```
* The Langchain Documents are later split into smaller chunks while preserving the context and meaning, which helps improve retrieval efficiency and reduce costs
* The splits are converted into embeddings (vector representations of words) and stored in the Vector Store. The Vector Stores are organized according to their respective categories
* The chat history is updated with the summary, and if the user provides a query along with an attachment, PDF retrieval takes care

#### LLM Router:
* It is an LLM that takes the user’s query and chat history to determine whether to use general knowledge for the answer or to inform the user that the query should be answered using the previously uploaded files. If the intent is unclear, it will indicate that as well
![image](https://github.com/user-attachments/assets/dcfaab2a-d23c-46de-a5cf-33d4bf097544)

#### PDF Retrieval:
![pdfRetrieval](https://github.com/user-attachments/assets/5101200e-c264-4876-9ff0-5280b32d7dc2)
* If the LLM Router determines that the query is related to previously uploaded files, the PDF retrieval process sends the query to the vector store retriever to obtain the relevant information
 ```
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 50, "filter": {'$and': [{'userid': {'$eq': userid}}, {'source': {'$eq': source}}]}})
# some filters can be used to get relevant information
````
* This relevant information is ranked based on the user’s query, and the top three ranked pieces of information are sent to the LLM along with the query to generate an answer
  
![image](https://github.com/user-attachments/assets/4811cd4f-f3f2-4987-80e6-6c36c37cf57c)

#### CSV Agent:
![csvAgent](https://github.com/user-attachments/assets/116cf42f-dbb0-48ff-b797-2ba38cef577b)

* First, the CSV file is downloaded and stored. A summary is generated that includes the column names and statistical information, which is then stored in the chat history
* A CSV agent is created with all the CSV files uploaded by the user, which is used to answer the query.



