# import streamlit as st
# from rag_chain import get_chain

# st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
# st.title("🤖 Simple RAG Chatbot")

# chain = get_chain([])

# user_input = st.chat_input("Type your message here...")

# if user_input:
#     with st.chat_message("user"):
#         st.write(user_input)

#     output = chain.invoke(user_input)

#     with st.chat_message("assistant"):
#         st.write(output)
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_chain

# 🧠 Set up the page with a chatbot theme
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = get_chain(st.session_state.chat_history)

# --- Display Previous Messages ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "ai"
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# --- User Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # 👤 Show user message
    st.chat_message("user", avatar="🧑").markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # 🤖 Generate chatbot response
    response = st.session_state.chain.invoke(user_input)

    # 🤖 Show chatbot response
    st.chat_message("ai", avatar="🤖").markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))