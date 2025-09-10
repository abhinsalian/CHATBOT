import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
apikey=os.getenv("GOOGLE_API_KEY")

def ingest_data(path="boook"):
    loader = DirectoryLoader(
        path=path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001',api_key=apikey)
    vector_store=Chroma(
        embedding_function=embedding,
        collection_name="sample",           
        persist_directory="./chatbot_db" 
    )

    vector_store.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks into Chroma DB")

if __name__ == "__main__":
    ingest_data("boook")