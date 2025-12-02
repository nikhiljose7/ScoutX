import os
import pandas as pd
# Monkeypatch can be removed or kept, it doesn't hurt, but we are removing google embeddings usage here.
# We'll keep imports clean.

from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# No need for GOOGLE_API_KEY for embeddings anymore
DATA_DIR = "moneyball_report_outputs"
FILES = [
    "all_predictions_with_undervaluation.csv",
    "players_data_cleaned_with_market_values_with_market_values1.csv"
]
INDEX_PATH = "faiss_index"

def create_vector_db():
    all_docs = []
    
    for filename in FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        print(f"Loading {filename}...")
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        docs = loader.load()
        
        # Add metadata about source
        for doc in docs:
            doc.metadata["source_file"] = filename
            
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} documents from {filename}")

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    print(f"Creating vector store with {len(all_docs)} documents using HuggingFace Embeddings...")
    # Use a standard, efficient model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store directly (no need for complex batching for rate limits)
    try:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        print(f"Saving index to {INDEX_PATH}...")
        vector_store.save_local(INDEX_PATH)
        print("Done.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    create_vector_db()
