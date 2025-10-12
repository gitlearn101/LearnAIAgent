## 1. Ingest PDF Files
# 2. Extract text from PDF files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings in a vector database
# 5. Perform similarity search on the vector database to find similar document
# 6. Retrieve the similar documents and present them to the user

## run pip install -r requirements.txt to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader    
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

doc_path = "data\JavaQB.pdf"
model = "llama3.2:latest"

# local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path = doc_path)
    data = loader.load()
    print("==== done loading... ====")

else:
    print("Upload a PDF file ..")    

# preview the first page of the PDF
#print(data[0].page_content[:500]) # print first 100 characters of the first page

# ===== End of PDF ingestion =====


# ===== Extract text from PDF files and split into small chunks =====
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 

# split and chunk the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("==== done splitting... ====")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk : {chunks[0].page_content}") # print first chunk


# ===== Add to vector DB =====
import ollama

ollama.pull("nomic-embed-text")  # pull the model if not already present

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="rag_java_QA",
      )

print("==== done adding to vector DB... ====")

# ===== Retrieval =====
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model
llm = ChatOllama(model=model) 

# simple technique to generate multiple queries from a single query
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to 
    Generate five different questions that are semantically similar to the question below. 
    Provide these alternative questions as a comma-separated list.
    \n\nQuestion: {question}\n\n1.""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT,
)

# RAG prompt
template = """Answer the question based ONLY on the following context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context" : retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input = ("Explain Java Memory Management?"))
print(res)
