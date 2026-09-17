# 📄 Chat with Your PDF (My RAG Search Engine)

Hi! Welcome to my semantic search project. I built this tool to make it easier to search through dense documents, reports, and textbooks. 

Instead of using `Ctrl + F` to find exact keyword matches, this app lets you upload a PDF and just ask questions in plain English. It reads the document and uses an AI model to generate a direct answer based purely on the text provided, so it doesn't make things up.

**Live Demo:** [Click here to try the app](https://pdf-rag-assistant-jlxtskg73ksra7c3qn6rde.streamlit.app/)

## 💡 Why I Built This

As a Statistics student at Carleton University, I deal with a lot of technical documents and data. I wanted to build a practical Retrieval-Augmented Generation (RAG) pipeline to get hands-on experience with vector databases and Large Language Models, while creating something genuinely useful. 

## 🚀 How It Works

1. **Upload:** You drop a PDF into the app.
2. **Process:** The app breaks the document down into small chunks and stores them in a local vector database.
3. **Search:** When you ask a question, it finds the most relevant chunks of text.
4. **Answer:** It sends those specific chunks to OpenAI to generate a clear, accurate answer (and even lets you see the exact source text it used!).

## 🛠️ What I Used to Build It

* **Python & Streamlit** for the frontend web interface.
* **LangChain** to connect all the pieces together.
* **OpenAI (gpt-4o-mini)** for the "brain" of the app.
* **ChromaDB** for storing and searching the document data.

## 💻 Want to run it on your own computer?

If you want to download my code and try running it locally, just open your terminal and follow these steps:

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/harissrinava/pdf-rag-assistant.git](https://github.com/harissrinava/pdf-rag-assistant.git)
   cd pdf-rag-assistant
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the app:**
   ```bash
   streamlit run app.py
   ```
*(Note: You'll need your own OpenAI API key to run it locally!)*

## 👨‍💻 About Me

**Haris Srinava**
* [Connect with me on LinkedIn](https://www.linkedin.com/in/haris-s-9b61131a8/)