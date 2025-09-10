from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
apikey=os.getenv("GOOGLE_API_KEY")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    return "\n".join(
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
        for msg in history
    )

def get_chain(chat_history):
    # load retriever from existing vector store
    embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001',api_key=apikey)
    vector_store=Chroma(
        embedding_function=embedding,
        collection_name="sample",           
        persist_directory="./chatbot_db" 
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    model=ChatGoogleGenerativeAI(model='gemma-3-27b-it',
                                api_key=apikey)

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

    # parallel inputs
    parallelchain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": RunnableLambda(lambda _: format_history(chat_history))
    })

    parser = StrOutputParser()
    return parallelchain | prompt | model | parser

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os

# load_dotenv()
# apikey=os.getenv("GOOGLE_API_KEY")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def format_history(history):
#     return "\n".join(
#         f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
#         for msg in history[-6:]  # keep last 6 exchanges for efficiency
#     )

# def get_chain(chat_history):
#     # embeddings
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=apikey)
#     vector_store = Chroma(
#         embedding_function=embedding,
#         collection_name="sample",
#         persist_directory="./chatbot_db"
#     )

#     # retriever with MMR
#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 6, "lambda_mult": 0.7}  # less redundancy, more relevance
#     )

#     # LLM
#     model = ChatGoogleGenerativeAI(
#         model="gemma-3-27b-it",
#         api_key=apikey,
#         temperature=0.2,
#         max_output_tokens=512
#     )

#     # better prompt
#     prompt = PromptTemplate(
#         template="""
#         You are a highly accurate assistant.
#         Answer based ONLY on the retrieved context and the conversation history.
#         - If the context does not contain the answer, say: "I don't know based on the provided information."
#         - Do not make up information.
#         - Keep answers clear, factual, and concise.

#         Conversation so far:
#         {history}

#         Retrieved Context:
#         {context}

#         User Question: {question}

#         Final Answer:
#         """,
#         input_variables=["history", "context", "question"]
#     )

#     # parallel inputs
#     parallelchain = RunnableParallel({
#         "context": retriever | RunnableLambda(format_docs),
#         "question": RunnablePassthrough(),
#         "history": RunnableLambda(lambda _: format_history(chat_history))
#     })

#     parser = StrOutputParser()

#     return parallelchain | prompt | model | parser
