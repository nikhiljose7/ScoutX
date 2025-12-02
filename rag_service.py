import os
from dotenv import load_dotenv
import google.generativeai.types as types

# Monkeypatch for GenerationConfig.MediaResolution error
# This must be done BEFORE importing langchain_google_genai
try:
    if not hasattr(types.GenerationConfig, 'MediaResolution'):
        class MediaResolution:
            UNSPECIFIED = 0
            LOW = 1
            MEDIUM = 2
            HIGH = 3
        types.GenerationConfig.MediaResolution = MediaResolution
except Exception:
    pass

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_PATH = "faiss_index"

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(INDEX_PATH):
            print(f"Warning: FAISS index not found at {INDEX_PATH}. RAG will not work.")
            return

        try:
            # Use the same model as in create_vector_db.py
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
            
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("RAG Service initialized successfully.")
        except Exception as e:
            print(f"Error initializing RAG Service: {e}")

    def get_answer(self, query):
        if not self.qa_chain:
            return "I'm sorry, but I can't access my knowledge base right now. Please try again later."
        
        try:
            result = self.qa_chain.invoke({"query": query})
            return result.get("result", "I couldn't generate an answer.")
        except Exception as e:
            return f"Error processing your request: {str(e)}"

# Singleton instance
rag_service = RAGService()
