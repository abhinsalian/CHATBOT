from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_chain
import os
import win32com.client  # Windows TTS

# Initialize Windows TTS
speaker = win32com.client.Dispatch("SAPI.SpVoice")

chat_history = []
chain = get_chain(chat_history)

print("🤖 Chatbot with Windows TTS is ready! Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("YOU: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Save user input
        chat_history.append(HumanMessage(content=user_input))

        # Get AI response
        output = chain.invoke(user_input)
        output = str(output)  # Ensure string

        # Print AI response
        print("AI:", output)

        # Save AI response
        chat_history.append(AIMessage(content=output)) 

        # Speak AI response using Windows TTS
        speaker.Speak(output)

    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")
        break

    except Exception as e:
        print("⚠️ Error:", e)
