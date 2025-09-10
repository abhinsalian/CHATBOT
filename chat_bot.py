from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.messages import HumanMessage,AIMessage


load_dotenv()

# api_key=os.getenv("GOOGLE_API_KEY")
 
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

loader=WebBaseLoader("https://rdltech.in")

docs=loader.load()

spliter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

chunks=spliter.split_documents(docs)

embeddings= GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vector_store=FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

chat_history = []

parser=StrOutputParser()

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

def format_history(history):
    return "\n".join(
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
        for msg in history
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallelchain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
    "history": RunnableLambda(lambda _: format_history(chat_history))
})

chain = parallelchain | prompt | model | parser

# while True:
#     user=input('YOU: ')
#     if user=='exit':
#         break
#     output=chain.invoke(user)
#     print('AI: ',output)

while True:
    user = input('YOU: ')
    if user.lower() == 'exit':
        break
    # if user.lower() == 'history':
    #     print(chat_history)
    
    # add user message to history
    chat_history.append(HumanMessage(content=user))
    
    # get output
    output = chain.invoke(user)
    print('AI:', output)
    
    # add AI response to history
    chat_history.append(AIMessage(content=output))

    # if user.lower() == "history":
    # for msg in chat_history:
    #     print(f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")
    # continue
