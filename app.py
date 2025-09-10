import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_chain

st.title("🤖 RAG Chatbot")

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

chat_history = []
chain = get_chain(chat_history)

print("🤖 Chatbot is ready! Type 'exit' to quit.\n")

user_input=st.text_input('Enter your promt')

if st.button('Answer'):
    output = chain.invoke(user_input)
    st.write(output)
    chat_history.append(AIMessage(content=output))