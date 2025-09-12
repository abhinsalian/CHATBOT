# from langchain_core.messages import HumanMessage, AIMessage
# from rag_chain import get_chain

# chat_history = []
# chain = get_chain(chat_history)

# print("🤖 Chatbot is ready! Type 'exit' to quit.\n")

# while True:    
#     user = input("YOU: ")
#     if user.lower() == "exit":
#         break

#     chat_history.append(HumanMessage(content=user))
#     output = chain.invoke(user)
#     print("AI:", output)
#     chat_history.append(AIMessage(content=output))
# import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_chain

chat_history = []
chain = get_chain(chat_history)

print("🤖 Chatbot is ready! Type 'exit' to quit.\n")

while True:    
    user = input("YOU: \n")
    if user.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=user))
    
    # Invoke the chain properly
    output=chain.invoke(user) # Use run() instead of invoke(user)
    
    # Print and save AI message
    print("AI:", output)
    chat_history.append(AIMessage(content=output))
