import streamlit as st
import os
import tempfile
import time
from io import BytesIO
from dotenv import load_dotenv

# LangChain & AI Imports
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION & THEME
st.set_page_config(
    page_title="MedStudy AI Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stChatMessage { border-radius: 12px; border: 1px solid #d1d8e0; padding: 15px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 8px; }
    .med-disclaimer { font-size: 0.8rem; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 20px; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. SESSION STATE INITIALIZATION
def init_state():
    defaults = {
        "messages": [],
        "current_file_hash": None,
        "retriever": None,
        "api_status": False,
        "processing_time": 0.0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# 3. UTILITY FUNCTIONS
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def check_api():
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY"))

# 4. DATA PIPELINE (The "Engine")
def ingest_document(file_bytes, file_name):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        
        loader = PyPDFLoader(path)
        raw_docs = loader.load()
        
        # Optimized Chunking for Medical Structures
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=250,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)
        
        # Create Vector Store
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=get_embeddings()
        )
        os.remove(path)
        return vector_db.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Ingestion Error: {str(e)}")
        return None

# 5. SIDEBAR DESIGN
with st.sidebar:
    st.title("⚕️ MedStudy AI Pro")
    st.info("System Status: **Active**" if check_api() else "System Status: **Missing API Key**")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📘 Upload Medical Manuscript", type="pdf")
    
    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file_hash != file_id:
            with st.status("🔬 Analyzing Document...") as status:
                st.session_state.retriever = ingest_document(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.current_file_hash = file_id
                st.session_state.messages = []
                status.update(label="Analysis Complete!", state="complete")

    st.markdown("### 🧠 Study Shortcuts")
    if st.button("📝 High-Yield Summary"):
        st.session_state.pending_query = "Summarize this document into 5 High-Yield bullet points for medical boards."
    if st.button("❓ Generate Quiz"):
        st.session_state.pending_query = "Create 3 clinical vignette-style MCQs based on this document with explained answers."
    
    st.markdown("---")
    if st.button("🗑️ Reset Session"):
        st.session_state.messages = []
        st.rerun()

    # Download Chat History
    if st.session_state.messages:
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Export Study Notes", data=chat_text, file_name="study_notes.txt")

    st.markdown("""
        <div class="med-disclaimer">
        <b>Disclaimer:</b> For educational purposes only. This AI can hallucinate. 
        Always cross-reference with primary clinical guidelines.
        </div>
        """, unsafe_allow_html=True)

# 6. MAIN CHAT LOGIC
if uploaded_file and st.session_state.retriever:
    # Model Setup
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.1, # Critical for medical accuracy
            max_tokens=2048
        )
    except Exception as e:
        st.error("LLM Connection Failed. Check your Internet/API Key.")

    # Optimized Medical Prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are an AI Medical Education Consultant.
    
    Source Material Context:
    {context}
    
    Strict Operating Instructions:
    1. If the answer is not contained within the Source Material Context, state: "Information not found in the provided manuscript."
    2. Maintain clinical tone. Use bullet points for lists.
    3. Do not infer patient dosages unless explicitly stated in the text.
    
    Student Question: {question}
    Clinical Response:""")

    # Display History
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input Handling
    user_input = st.chat_input("Ask a clinical or research question...")
    final_query = user_input or st.session_state.pop("pending_query", None)

    if final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        st.chat_message("user").markdown(final_query)

        with st.chat_message("assistant"):
            try:
                start_time = time.time()
                # 1. Retrieve
                with st.spinner("Searching manuscript..."):
                    docs = st.session_state.retriever.invoke(final_query)
                    context = "\n\n".join([d.page_content for d in docs])
                
                # 2. Generate
                with st.spinner("Synthesizing response..."):
                    chain = prompt_template | llm | StrOutputParser()
                    response = chain.invoke({"context": context, "question": final_query})
                    st.session_state.processing_time = round(time.time() - start_time, 2)
                
                # 3. Render
                st.markdown(response)
                st.caption(f"⏱️ Response Time: {st.session_state.processing_time}s")
                
                with st.expander("📚 View Evidence Chunks"):
                    for i, d in enumerate(docs):
                        st.write(f"**Snippet {i+1} (Page {d.metadata.get('page', 0)+1}):**")
                        st.info(d.page_content)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    # Landing Page
    st.markdown("""
        ## Welcome to MedStudy AI Pro
        To begin, upload a medical PDF (Research paper, textbook chapter, or clinical notes) in the sidebar.
        
        **How it works:**
        1. **Vectorization:** We convert your PDF into semantic 'embeddings'.
        2. **RAG Pipeline:** When you ask a question, we pull only the relevant facts.
        3. **Llama-3 Integration:** A medical-tuned prompt ensures accurate, professional answers.
    """)
    st.image("https://www.medmastery.com/sites/default/files/styles/extra_large/public/2020-04/illustration-medical-education.png", width=500)