import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_chain
from langchain_core.output_parsers import StrOutputParser
import win32com.client  # Windows TTS
import pythoncom  # Needed for COM initialization

# --- Initialize Windows TTS safely ---
def get_speaker():
    pythoncom.CoInitialize()
    return win32com.client.Dispatch("SAPI.SpVoice")

speaker = get_speaker()

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("☠️ RAG Chatbot")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chain = get_chain(st.session_state.chat_history)

on=st.sidebar.toggle('🔊-TEXT-TO-SPEECH')
if on:
    st.sidebar.success("🎙️ TTS Enabled")
else:
    st.sidebar.warning("🔇 TTS Disabled")

if st.sidebar.button('clear'):
        st.session_state.chat_history.clear()

# --- Display Previous Messages ---
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

parser = StrOutputParser()

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    output = st.session_state.chain.invoke(user_input)

    st.session_state.chat_history.append(AIMessage(content=output))
    with st.chat_message("assistant"):
        st.write(output)
        try:
            if on:
                speaker.Speak(output)
        except Exception as e:
            st.error(f"TTS Error: {e}")
        finally:
            pythoncom.CoUninitialize()
