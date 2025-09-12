# 🤖 Chatbot Project

A **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Streamlit**, **Python**, and **Hugging Face**.

---

## 📂 Project Structure

```
chatbot/
│── app.py              # Streamlit UI
│── main.py             # Main script
│── rag_chain.py        # RAG pipeline
│── data_ingestion.py   # Data ingestion scripts
│── doc_loader.py       # Document loader UI
│── text_to_speech.py   # Text-to-speech
│── chatbot_db/         # Vector database
│── Panchatantra.pdf    # Example document
│── README.md           # Project instructions
```

---

## ⚙️ Setup

### 1️⃣ Environment Variables  
Create a **`.env`** file in the root directory and add your API keys:  

```env
GOOGLE_API_KEY=your_api_key
```

---

## ▶️ Running the App

### 🔹 UI Mode (Streamlit)  
Run these scripts for the chatbot with a UI:  

```bash
streamlit run doc_loader.py
python rag_chain.py
streamlit run app.py
```

### 🔹 Normal Python Mode  
Run these scripts for the terminal-based chatbot:  

```bash
python data_ingestion.py
python rag_chain.py
python main.py
```

---

✅ **Note:**  
- You do **not** need to run `rag_chain.py` directly.  
- It’s automatically imported by `main.py` and `app.py`.