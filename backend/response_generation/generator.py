import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import Optional
from better_profanity import profanity
from backend.workers.utilities import partial_censor

# Global variables for model and tokenizer
model = None
tokenizer = None
rag_chain = None
device = None

profanity.load_censor_words()
# Load config file
with open("config.json", "r") as f:
    config = json.load(f)
max_new_tokens = config.get("max_new_tokens", 512)  # Default to 512 if not set
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
        from backend.rag.retriever_manager import get_retrievers_with_auto_reload
        
        # Get retrievers with automatic reload check
        faiss_db, hybrid_retriever = get_retrievers_with_auto_reload()
        
        if hybrid_retriever is None:
            print("⚠️ Retrievers not found after reload check")
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

def refresh_rag_chain():
    """Force refresh of RAG chain with updated retrievers"""
    global rag_chain
    
    try:
        from backend.rag.retriever_manager import get_retrievers_with_auto_reload
        
        print("🔄 Refreshing RAG chain with updated retrievers...")
        
        # Get fresh retrievers with auto-reload
        faiss_db, hybrid_retriever = get_retrievers_with_auto_reload()
        
        if hybrid_retriever is None:
            print("❌ Failed to get updated retrievers")
            return False
        
        # Initialize model if not already done
        initialize_model()
        
        # Rebuild RAG chain with new retrievers
        custom_llm = LambdaLLM()
        rag_chain = RetrievalQA.from_chain_type(
            llm=custom_llm,
            retriever=hybrid_retriever,
            return_source_documents=True
        )
        
        print("✅ RAG chain refreshed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to refresh RAG chain: {e}")
        return False

# Prompt formatting (this is where prompt engineering happens)
def format_prompt(question: str, context: str) -> str:
    with open("context.txt", "a") as f:
        f.write(f"\n\nQuestion: {question}\n")
        f.write(f"Context: {context}\n")
    return f"""You are an intelligent assistant capable of holding conversations. Your response should be concise, informative, and relevant to the user's questions. Ignore the context if it  is not usefull. Answer within {max_new_tokens*4-100} characters.

    Question: {question}

    Context:
    {context}

    Answer:"""

def format_prompt_no_rag(question: str) -> str:
    """Format prompt for non-RAG responses"""
    with open("context.txt", "a") as f:
        f.write(f"\n\nQuestion (No RAG): {question}\n")
    return f"""You are an intelligent assistant capable of holding conversations. Your response should be concise, informative, and relevant to the user's questions. Answer within {max_new_tokens*4-100} characters.

    Question: {question}

    Answer:"""

# Custom LLM wrapper for LangChain
class LambdaLLM(LLM):
    def _call(self, prompt: str, stop=None):
        return generate_text(prompt)
    
    @property
    def _llm_type(self) -> str:
        return "lambda"

def generate_text(prompt: str) -> str:
    """Generate text using the loaded model"""
    initialize_model()  # Ensure model is loaded
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (response)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def generate_response(query: str, use_rag: bool = True, fallback_on_error: bool = True, user_id: str = None, tenant_id: str = None) -> str:
    """
    Generate response with optional RAG and fallback capabilities
    Now includes automatic retriever reload checking
    """
    try:
        if use_rag:
            # Initialize RAG chain (will auto-reload if needed)
            initialize_rag_chain()
            
            # Check if we need to refresh the RAG chain due to retriever updates
            from backend.rag.retriever_manager import retriever_manager
            if retriever_manager.should_reload_retrievers():
                print("🔄 Detected retriever updates, refreshing RAG chain...")
                if refresh_rag_chain():
                    print("✅ RAG chain refreshed with updated retrievers")
                else:
                    print("❌ Failed to refresh RAG chain, using existing one")
            
            if rag_chain is None:
                raise ValueError("RAG chain not initialized")
            
            print(f"🔍 Using RAG for query: {query[:50]}...")
            
            # Use RAG chain
            result = rag_chain({"query": query})
            response = result.get("result", "").strip()
            
            if response:
                print(f"✅ RAG response generated: {len(response)} chars")
                return response
            else:
                print("⚠️ RAG returned empty response")
                if fallback_on_error:
                    print("🔄 Falling back to non-RAG response...")
                    return generate_response(query, use_rag=False, fallback_on_error=False, user_id=user_id, tenant_id=tenant_id)
                return "I couldn't find relevant information to answer your question."
        
        else:
            # Non-RAG response
            initialize_model()  # Ensure model is loaded
            print(f"🤖 Using non-RAG for query: {query[:50]}...")
            
            prompt = format_prompt_no_rag(query)
            response = generate_text(prompt)
            
            if response:
                print(f"✅ Non-RAG response generated: {len(response)} chars")
                return response
            else:
                return "I'm having trouble generating a response right now."
    
    except Exception as e:
        print(f"❌ Error in generate_response: {e}")
        
        if use_rag and fallback_on_error:
            print("🔄 RAG failed, falling back to non-RAG...")
            try:
                return generate_response(query, use_rag=False, fallback_on_error=False, user_id=user_id, tenant_id=tenant_id)
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
        
        return "I'm experiencing technical difficulties. Please try again later."

# Test functions (for development)
def test_response_generation():
    print("\n\n")
    # Test query
    query = "If you don't know the answer just say so. here is my question: What is the dress code for students at FAST University?"
    print("Query:", query)
    result = rag_chain.invoke({"query": query})
    result = partial_censor(result["result"]) if result and "result" in result else "No answer found."
    print("💬 Answer:", result)
    print("\n")
    
    query = "Answer this in urdu. If you don't know the answer just say that you don't know. Question: اگر آپ کو جواب نہیں معلوم تو بس اتنا کہہ دیں۔ میرا سوال یہ ہے: فاسٹ یونیورسٹی میں طلباء کے لیے ڈریس کوڈ کیا ہے؟"
    print("Query:", query)
    result = partial_censor(result["result"]) if result and "result" in result else "No answer found."
    print("💬 Answer:", result)

# For Celery worker initialization
def initialize_for_celery():
    """Initialize all components for Celery worker"""
    try:
        initialize_model()
        initialize_rag_chain()
        print("✅ All components initialized for Celery worker")
    except Exception as e:
        print(f"❌ Failed to initialize components for Celery: {e}")
        # Still initialize the model so we can use fallback mode
        try:
            initialize_model()
            print("✅ Model initialized for fallback mode")
        except Exception as model_error:
            print(f"❌ Failed to initialize model: {model_error}")
