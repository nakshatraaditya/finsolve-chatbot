import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vectorstores/db/"

def create_vector_db():
    """
    Loads documents from various subfolders and file types, assigns department
    metadata, creates embeddings, and persists them in a Chroma vector database.
    """
    print("--- Starting data ingestion process ---")

    # Check if the data path exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: The data directory '{DATA_PATH}' was not found.")
        print("Please make sure you are running this script from the root of the 'finsolve-chatbot' project folder.")
        return

    all_documents = []
    department_folders = os.listdir(DATA_PATH)

    if not department_folders:
        print(f"❌ ERROR: The data directory '{DATA_PATH}' is empty.")
        print("Please make sure it contains the department subfolders (Finance, HR, etc.).")
        return
    
    print(f"Found {len(department_folders)} department folders.")

    # Iterate through each department subfolder in the data directory
    for department in department_folders:
        department_path = os.path.join(DATA_PATH, department)
        if not os.path.isdir(department_path):
            continue

        print(f"\n➡️ Processing folder: {department_path}")
        
        # Define loaders for different file types within the department folder
        txt_loader = DirectoryLoader(department_path, glob="**/*.md", loader_cls=TextLoader, show_progress=False)
        pdf_loader = DirectoryLoader(department_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
        csv_loader = DirectoryLoader(department_path, glob="**/*.csv", loader_cls=CSVLoader, show_progress=False)
        
        loaded_documents = txt_loader.load() + pdf_loader.load() + csv_loader.load()

        if not loaded_documents:
            print(f"   ⚠️ No documents found in this folder.")
            continue
            
        print(f"   ✅ Loaded {len(loaded_documents)} document(s).")
        
        # Attach the department name as metadata to each document
        for doc in loaded_documents:
            doc.metadata["department"] = department
        
        all_documents.extend(loaded_documents)
    
    if not all_documents:
        print("\n❌ ERROR: No documents were loaded in total. Please check that your department folders contain .md, .pdf, or .csv files.")
        return

    # Split documents into smaller chunks
    print("\nSplitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    texts = text_splitter.split_documents(all_documents)
    print(f"   Split into {len(texts)} chunks.")

    # Create embeddings
    print("\nCreating vector embeddings...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist the vector store
    print("\nCreating and persisting the vector database...")
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embedding_function, 
        persist_directory=DB_PATH
    )

    print("\n\n✅✅✅ Vector database created and populated successfully! ✅✅✅")


if __name__ == "__main__":
    create_vector_db()