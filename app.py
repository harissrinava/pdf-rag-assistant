import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDF Semantic Search", page_icon="📄")
st.title("📄 Chat with Your PDF")

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.info("Please add your OpenAI API key in the sidebar to continue.")
    st.stop()
    
os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    with st.spinner("Processing document..."):
        loader = PyMuPDFLoader(tmp_filepath)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        if len(splits) == 0:
            st.error("Could not extract any text. This PDF might be a scanned image.")
            st.stop()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.success("Document ready!")

    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_question})
            st.markdown("### Answer:")
            st.write(response["answer"])
            
            with st.expander("View Source Text"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)