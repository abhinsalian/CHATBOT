# 🤖 Chatbot Project
A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit,Python and Hugging Face.

## 📂 Project Structure
chatbot/
│── app.py # Streamlit UI
│── main.py # Main script
│── rag_chain.py # RAG pipeline
│── data_ingestion.py # Data ingestion scripts
│── doc_loader.py # Document loader UI
│── text_to_speech.py # Text-to-speech
│── chatbot_db/ # Vector database
│── Panchatantra.pdf # Example document
│── README.md # Project instructions

Set up environment variables
Create a .env file in the root with your API keys:
GOOGLE_API_KEY=your_api_key

## ▶️ Running the App
🔹 UI Mode (Streamlit)
Run these scripts for the chatbot with UI:
streamlit run doc_loader.py
python rag_chain.py
streamlit run app.py

🔹 Normal Python Mode
Run these scripts for the terminal-based chatbot:
python data_ingestion.py
python rag_chain.py
python main.py