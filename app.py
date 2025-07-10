import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinSolve - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" # Keep the sidebar open initially
)

# --- APP TITLE AND DESCRIPTION ---
st.title("FinSolve Technologies AI Assistant 🤖")

# --- RBAC PERMISSIONS ---
RBAC_PERMISSIONS = {
    "Finance": ["Finance", "General"],
    "Marketing": ["Marketing", "Finance", "General"],
    "HR": ["HR", "General"],
    "Engineering": ["Engineering", "General"],
    "C-Level": ["Finance", "Marketing", "HR", "Engineering", "General"],
    "Employee": ["General"]
}

# --- CACHING FUNCTIONS FOR PERFORMANCE ---
@st.cache_resource
def load_llm(api_key):
    """Loads the language model, caching it for performance."""
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)

@st.cache_resource
def load_vector_store():
    """Loads the vector store, caching it for performance."""
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Make sure the path to your vector store is correct
    if not os.path.exists("vectorstores/db/"):
        st.error("Vector store not found! Please run ingest.py first.")
        st.stop()
    return Chroma(persist_directory="vectorstores/db/", embedding_function=embedding_function)

# --- LOAD RESOURCES AND API KEY ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    llm = load_llm(google_api_key)
    vectorstore = load_vector_store()
except (FileNotFoundError, KeyError):
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY secret in Streamlit Cloud.")
    st.stop()

PROMPT = PromptTemplate(
    template="""
    You are an expert AI assistant for FinSolve Technologies. Answer questions accurately based ONLY on the provided context. If the information is not present, state that you do not have access to that information. Do not make up facts.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """, 
    input_variables=["context", "question"]
)

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.header("👤 User Controls")
    selected_role = st.selectbox("Select Your Role", list(RBAC_PERMISSIONS.keys()))
    st.info(f"Logged in as: **{selected_role}**")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("ℹ️ About This App & Role Permissions"):
        st.markdown("""
            This chatbot uses a Role-Based Access Control (RBAC) system. Your selected role determines which documents you can access.
            - **C-Level**: Full access.
            - **Marketing**: Access to Marketing, Finance, and General data.
            - **Finance**: Access to Finance and General data.
            - **HR**: Access to HR and General data.
            - **Employee**: Access to General company info only.
        """)

# --- CHAT INTERFACE ---

# Initialize or display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please select your role and ask a question."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question based on your role..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # --- RAG and RBAC Logic ---
                allowed_departments = RBAC_PERMISSIONS[selected_role]
                metadata_filter = {"department": {"$in": allowed_departments}}
                
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 5, 'filter': metadata_filter}
                )
                
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                
                if not context:
                    full_response = "I do not have access to the information required to answer your question. Please check your permissions."
                    st.markdown(full_response)
                else:
                    formatted_prompt = PROMPT.format(context=context, question=prompt)
                    
                    # Use the stream for a more interactive feel
                    response_stream = llm.stream(formatted_prompt)
                    full_response = st.write_stream(response_stream)
            
            # Append full response to history after streaming is complete
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            # Catch potential errors from the API or other issues
            st.error(f"An error occurred: {e}")
            error_message = "Sorry, I ran into a problem. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": error_message})