from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader    
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "data/JavaQB.pdf"
MODEL_NAME = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "rag_java_QA"

def ingest_pdf(doc_path):
    """ Load PDF document """
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file does not exist > {doc_path}")
        return None
    
def split_documents(documents):
    """ Split documents into chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def create_vector_db(chunks):
    """ Create vector database from document chunks """
    import ollama
    ollama.pull(EMBEDDING_MODEL)  # Ensure the embedding model is available

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db

def create_retriever(vector_db, llm):
    """ Create a multi-query retriever """
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
    logging.info("Multi-query retriever created.")
    return retriever

def create_chain(retriever, llm):
    """ Create the RAG chain """
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
    logging.info("RAG chain created.")
    return chain

def main():
    st.title("PDF Assistant - PDF RAG with Streamlit and Ollama")
    st.write(":green[PDF is already loaded. Enter your query below.]")

    # Load and ingest PDF (already present)
    data = ingest_pdf(DOC_PATH)
    if data is None:
        st.error(f"PDF file not found: {DOC_PATH}")
        return

    # Split documents into chunks
    chunks = split_documents(data)

    # Create vector database
    vector_db = create_vector_db(chunks)

    # Set up LLM
    llm = ChatOllama(model=MODEL_NAME)

    # Create retriever
    retriever = create_retriever(vector_db, llm)

    # Create RAG chain
    chain = create_chain(retriever, llm)

    # UI: Only show a textbox for user query
    query = st.text_input("Enter your question about the PDF:")

    if query:
        with st.spinner("Generating answer..."):
            response = chain.invoke(input=query)
            st.success("Answer:")
            st.write(response)


if __name__ == "__main__":
    main()
