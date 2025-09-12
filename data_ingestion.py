from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
 
# def ingest_data(path="boook"):
#     loader = DirectoryLoader(
#         path=path,
#         glob="*.pdf",
#         loader_cls=PyPDFLoader
#     )
#     docs = loader.load()

def ingest_data(path="Panchatantra.pdf"):
    loader = PyPDFLoader(path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)
    
    #embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=api_key)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L12-v2"
    )

    vector_store = Chroma(
        embedding_function=embedding,
        collection_name="sample",
        persist_directory="./chatbot_db"
    )

    vector_store.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks into Chroma DB with EmbeddingGemma")

if __name__ == "__main__":
    ingest_data("boook")