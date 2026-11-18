import streamlit as st
import tempfile
import os
import uuid
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDF Semantic Search", page_icon="📄")
st.title("📄 Chat with Your PDF")

# Check Streamlit Cloud Secrets first, fall back to manual sidebar input
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

# 1. Initialize user session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex}"
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "vector_retriever" not in st.session_state:
    st.session_state.vector_retriever = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # 2. Only index if it's a new or different file for this user session
    if st.session_state.current_file != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name

        with st.spinner("Processing document..."):
            loader = PyMuPDFLoader(tmp_filepath)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            if len(splits) == 0:
                st.error("Could not extract any text. This PDF might be a scanned image.")
                st.stop()

            st.session_state.num_chunks = len(splits)
            embeddings = OpenAIEmbeddings()

            # Isolate this user's vectors into their own unique Chroma collection
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=st.session_state.session_id
            )
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # Initialize lexical BM25 retriever and combine into Hybrid EnsembleRetriever
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 5

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )

            st.session_state.vector_retriever = vector_retriever
            st.session_state.bm25_retriever = bm25_retriever

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
            st.session_state.rag_chain = create_retrieval_chain(hybrid_retriever, question_answer_chain)
            st.session_state.current_file = uploaded_file.name

            st.success("Document ready!")

    # 3. Chat Interface using the isolated session chain
    if st.session_state.rag_chain is not None:
        user_question = st.text_input("Ask a question about the document:")
        
        if user_question:
            with st.spinner("Thinking..."):
                start_time = time.perf_counter()
                response = st.session_state.rag_chain.invoke({"input": user_question})
                latency_ms = (time.perf_counter() - start_time) * 1000

                st.markdown("### Answer:")
                st.write(response["answer"])
                
                with st.expander("View Source Text"):
                    for i, doc in enumerate(response["context"]):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {i+1} (Page {page_num}):**")
                        st.write(doc.page_content)

                # 4. Lightweight Expander Backing Resume Hybrid Retrieval & Telemetry Claims
                with st.expander("🔍 View Hybrid Retrieval Benchmark (BM25 + Chroma Vector) & Telemetry"):
                    vec_docs = st.session_state.vector_retriever.invoke(user_question)
                    bm25_docs = st.session_state.bm25_retriever.invoke(user_question)
                    
                    vec_pages = {d.metadata.get("page", 0) + 1 for d in vec_docs[:3]}
                    bm25_pages = {d.metadata.get("page", 0) + 1 for d in bm25_docs[:3]}
                    hybrid_pages = {d.metadata.get("page", 0) + 1 for d in response["context"][:3]}

                    c1, c2, c3 = st.columns(3)
                    c1.metric("End-to-End Query Latency", f"{latency_ms:,.0f} ms")
                    c2.metric("Indexed Document Chunks", f"{st.session_state.num_chunks}")
                    c3.metric("Hybrid Top-3 Source Pages", f"Pages {sorted(list(hybrid_pages))}")

                    st.caption(
                        f"**Retriever Comparison (Top-3 Pages):** "
                        f"Dense Vector Only: `{sorted(list(vec_pages))}` | "
                        f"BM25 Lexical Only: `{sorted(list(bm25_pages))}` | "
                        f"Reciprocal Rank Fusion (0.6 Vector / 0.4 BM25): `{sorted(list(hybrid_pages))}`"
                    )