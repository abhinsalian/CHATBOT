import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader,TextLoader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import streamlit as st
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
apikey=os.getenv("GOOGLE_API_KEY")

st.title("📁 Upload File")
uploaded_file = st.file_uploader("Choose a file")

def ingest_data(path):

    if uploaded_file.type == 'application/pdf':
        loader=PyPDFLoader(file_path=path)
    elif uploaded_file.type == 'text/plain':
        loader=TextLoader(file_path=path,encoding='utf-8')
    elif uploaded_file.type == 'text/csv':
        loader=CSVLoader(file_path=path)
    else:
        st.error("Unsupported file type")
        return
        
    docs=loader.load()

    if st.button('Split & Ingest'):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001',api_key=apikey)
        embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L12-v2"
        )
        vector_store=Chroma(
            embedding_function=embedding,
            collection_name="sample",           
            persist_directory="./chatbot_db" 
        )
        vector_store.add_documents(chunks)

        print(f"Ingested {len(chunks)} chunks into chatbot_db")
        st.success(f"Ingested {len(chunks)} chunks into chatbot_db")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File type:", uploaded_file.type)
    ingest_data(path=uploaded_file.name)