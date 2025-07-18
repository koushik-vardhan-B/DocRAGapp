# Documents RAG Application

This project is a Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to query research papers using LLMs and embeddings. It leverages LangChain, FAISS, and open-source models for document search and question answering.

## Features

- Upload and index PDF research papers for semantic search
- Query documents using natural language
- Uses Llama3 via Groq API for answering questions
- Embeddings powered by Ollama (`nomic-embed-text`)
- Vector database built with FAISS
- Document similarity search and context display

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r ../requirements.txt
   ```

2. **Set up environment variables:**
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY="your_groq_api_key"
     ```

3. **Add your research papers:**
   - Place PDF files in the `research_papers/` directory.

4. **Run the app:**
   ```sh
   streamlit run app.py
   ```

## Usage

- Click "Document Embedding" to build the vector database from your PDFs.
- Enter your query in the input box.
- View the answer and relevant document context.

## File Structure

- [`app.py`](app.py): Main Streamlit application.
- [`research_papers/`](research_papers/): Directory for PDF documents.
- [`requirements.txt`](../requirements.txt): Python dependencies.
- [`.env`](../.env): Environment variables (not committed to GitHub).

## License

MIT License