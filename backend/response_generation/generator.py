import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import Optional

# Global variables for model and tokenizer
model = None
tokenizer = None
rag_chain = None
device = None

# ---------------- Setup ------------------
def initialize_model():
    """Initialize model and tokenizer - called by Celery worker or on first use"""
    global model, tokenizer, device
    
    if model is not None and tokenizer is not None:
        return  # Already initialized
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Initializing model on {device}...")
    
    with open("config.json", "r") as f:
        config = json.load(f)

    # Model & token setup
    model_id = config["mistral_model_id"][device]
    max_new_tokens = config["max_new_tokens"]
    
    # Load model/tokenizer (causal model now)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    print(f"✅ Model {model_id} loaded successfully on {device}")

def initialize_rag_chain():
    """Initialize RAG chain - called by Celery worker or on first use"""
    global rag_chain
    
    if rag_chain is not None:
        return  # Already initialized
    
    try:
        from backend.rag.retrieverconstruction import get_retrievers, construct_retrievers
        
        # Get retrievers (they should be initialized by Celery worker)
        faiss_db, hybrid_retriever = get_retrievers()
        
        if hybrid_retriever is None:
            print("⚠️ Retrievers not found, constructing new ones...")
            faiss_db, hybrid_retriever = construct_retrievers()
        
        if hybrid_retriever is None:
            raise ValueError("Failed to initialize retrievers")
        
        # Initialize model if not already done
        initialize_model()
        
        # Build RAG chain
        custom_llm = LambdaLLM()
        rag_chain = RetrievalQA.from_chain_type(
            llm=custom_llm,
            retriever=hybrid_retriever,
            return_source_documents=True
        )
        
        print("✅ RAG chain initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG chain: {e}")
        raise

# Prompt formatting (this is where prompt engineering happens)
def format_prompt(question: str, context: str) -> str:
    with open("context.txt", "a") as f:
        f.write(f"\n\nQuestion: {question}\n")
        f.write(f"Context: {context}\n")
    return f"""You are an intelligent assistant capable of holding conversations. Your response should be concise, informative, and relevant to the user's questions. If the question is not related to the context, ignore the context. Answer in a smooth paragraph format. 

    Context:
    {context}
    
    Question: {question}
    Answer:"""

# LLM wrapper
def call_llm(question: str, context: str) -> str:
    """Call the LLM with question and context"""
    global model, tokenizer, device
    
    # Initialize if needed
    if model is None or tokenizer is None:
        initialize_model()
    
    with open("config.json", "r") as f:
        config = json.load(f)
    max_new_tokens = config["max_new_tokens"]
    
    # Use the format_prompt function to create the final prompt
    prompt = format_prompt(question, context)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

class LambdaLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        # Here we pass the context and question to the call_llm function
        context = ""  # This will be handled by the RAG chain
        return call_llm(prompt, context)

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

def generate_response(query: str) -> str:
    """
    Generate a response to the given query using the RAG chain.
    Thread-safe for Celery workers.
    """
    global rag_chain
    
    # Initialize if needed (for non-Celery usage)
    if rag_chain is None:
        initialize_rag_chain()
    
    try:
        result = rag_chain.invoke({"query": query})
        return result["result"] if "result" in result else "No answer found."
    except Exception as e:
        print(f"❌ Error in generate_response: {e}")
        return f"Error generating response: {str(e)}"

def generate_response_with_reranking(query: str, top_k: int = 2) -> str:
    """
    Generate a response using RAG with reranking.
    Thread-safe for Celery workers.
    """
    try:
        from backend.rag.retrieverconstruction import get_retrievers
        from backend.response_generation.reranker import rerank_docs
        
        # Get retrievers
        faiss_db, hybrid_retriever = get_retrievers()
        
        if hybrid_retriever is None:
            return "Error: Retrievers not initialized"
        
        # Initialize model if needed
        if model is None or tokenizer is None:
            initialize_model()
        
        # Retrieve initial documents
        docs = hybrid_retriever.get_relevant_documents(query)
        
        # Rerank the documents
        reranked_docs = rerank_docs(query, docs, top_k=top_k)
        
        # Generate response using the reranked documents
        context = "\n".join([doc.page_content for doc in reranked_docs])
        return call_llm(query, context)
        
    except Exception as e:
        print(f"❌ Error in generate_response_with_reranking: {e}")
        return f"Error generating response with reranking: {str(e)}"

# Test functions (for development)
def test_response_generation():
    print("\n\n")
    # Test query
    query = "If you don't know the answer just say so. here is my question: What is the dress code for students at FAST University?"
    print("Query:", query)
    result = rag_chain.invoke({"query": query})
    print("💬 Answer:", result["result"])
    
    print("\n")
    query = "Answer this in urdu. If you don't know the answer just say that you don't know. Question: اگر آپ کو جواب نہیں معلوم تو بس اتنا کہہ دیں۔ میرا سوال یہ ہے: فاسٹ یونیورسٹی میں طلباء کے لیے ڈریس کوڈ کیا ہے؟"
    print("Query:", query)
    result = rag_chain.invoke({"query": query})
    print("💬 Answer:", result["result"])

# For Celery worker initialization
def initialize_for_celery():
    """Initialize all components for Celery worker"""
    print("🔧 Initializing response generation for Celery worker...")
    try:
        initialize_model()
        initialize_rag_chain()
        print("✅ Response generation initialized for Celery")
    except Exception as e:
        print(f"❌ Failed to initialize response generation for Celery: {e}")
        raise
