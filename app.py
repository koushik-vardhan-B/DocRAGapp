import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load .env and API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

# Set up the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    <context>
    
    Question: {input}
    """
)

# Embedding & vector creation function
def create_vectors_embedding():
    if "vectors" not in st.session_state:
        if not os.path.exists("research_papers"):
            st.error("❌ 'research_papers' folder not found. Please add your PDFs.")
            st.stop()
        
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("📚 Chat with Research Papers using RAG + Groq")
user_prompt = st.text_input("Enter your query from the documents")

# Embedding button
if st.button("Document Embedding"):
    with st.spinner("Creating vector database..."):
        create_vectors_embedding()
    st.success("✅ Vector database is ready!")

# Query the vector DB
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please generate the document embeddings first by clicking 'Document Embedding'.")
    else:
        # Retrieve documents and run QA chain
        docs = st.session_state.vectors.similarity_search(user_prompt, k=3)
        document_chain = create_stuff_documents_chain(llm, prompt)

        start = time.process_time()
        response = document_chain.invoke({"input": user_prompt, "context": docs})
        duration = time.process_time() - start

        st.write(f"**🧠 Assistant:** {response['answer']}")
        st.write(f"⏱️ Response time: {duration:.2f} seconds")

        with st.expander("📄 Similar Documents"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
