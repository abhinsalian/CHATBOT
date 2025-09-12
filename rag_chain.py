from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import torch

load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    return "\n".join(
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
        for msg in history
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
def get_chain(chat_history):
    #embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001',api_key=apikey)
    # embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
    )

    vector_store = Chroma(
        embedding_function=embedding,
        collection_name="sample",
        persist_directory="./chatbot_db"
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    model = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        api_key=apikey
    )

    # prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Use the chat history and the provided context to answer the question.
        If the context is insufficient, just say you don't know.

        Chat History:
        {history}

        Context:
        {context}

        Question: {question}
        """,
        input_variables=['history', 'context', 'question']
    )

    parallelchain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": RunnableLambda(lambda _: format_history(chat_history))
    })

    parser = StrOutputParser()
    return parallelchain | prompt | model | parser