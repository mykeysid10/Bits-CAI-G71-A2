"""
Retrieval-Augmented Generation (RAG) system for financial Q&A.
Uses lightweight alternatives to NLTK for tokenization and stopwords.
"""

import os
import torch
import time
import pickle
import faiss
import string
import re
import warnings
from typing import Dict, Tuple, List
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

# Lightweight stopwords list (English)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't"
}

class RAGGuardrails:
    """Input validation and guardrails for RAG system."""
    
    def __init__(self):
        """Initialize harmful patterns and responses."""
        self.harmful_patterns = {
            "violence": ["kill", "attack", "shoot", "bomb"],
            "financial_crime": ["launder", "fraud", "scam"],
            "personal_info": ["ssn", "credit card", "password"],
            "greetings": ["hi", "hello", "hey", "how are you", "what's up"],
            "out_of_scope": ["capital of", "france", "weather", "sports", "movie"]
        }
    

    def validate_query(self, query: str) -> Tuple[bool, str, str]:
        """Check for harmful or greeting queries."""
        query_lower = query.lower().strip()
        
        if any(pattern in query_lower for pattern in self.harmful_patterns["greetings"]):
            return False, "Hello! I'm a financial Q&A assistant. Please ask me about financial statements.", "greeting"
        
        for category, patterns in self.harmful_patterns.items():
            if category == "greetings":
                continue
            if any(pattern in query_lower for pattern in patterns):
                if category == "out_of_scope":
                    return False, "This question is outside my financial domain expertise.", "out_of_scope"
                return False, f"I cannot answer questions related to {category.replace('_', ' ')}.", "guardrail"
        
        if len(query_lower.split()) < 3:
            return False, "Please provide a more detailed question.", "short_query"
            
        return True, "", "valid"


class RAGSystem:
    """Main RAG system implementation."""
    
    # In backend_app.py - RAGSystem __init__ method
    def __init__(self, artifacts_dir="rag-artifacts"):
        self.artifacts_dir = artifacts_dir
        self.guardrails = RAGGuardrails()
        print(f"🔄 Initializing RAG System with artifacts from: {artifacts_dir}")
        self.load_models()
        self.qa_model = self.initialize_qa_model()
        print("✅ RAG System initialized successfully")

    def load_models(self):
        """Load pre-built models and indexes."""
        print("📦 Loading RAG models and indexes...")
        start_time = time.time()
        
        os.makedirs('local_models', exist_ok=True)
        
        # Get device first
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Device set to: {device}")
        
        # Load model directly to device
        print("🔌 Loading embedding model...")
        self.embed_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            cache_folder='local_models',
            device=device
        )
        print("✅ Embedding model loaded")
        
        print("📊 Loading FAISS index...")
        self.faiss_index = faiss.read_index(os.path.join(self.artifacts_dir, "faiss_index.index"))
        print("✅ FAISS index loaded")
        
        print("📋 Loading BM25 index...")
        with open(os.path.join(self.artifacts_dir, "bm25_index.pkl"), "rb") as f:
            self.bm25_index, self.chunks = pickle.load(f)
        print("✅ BM25 index loaded")
        
        print("⚖️ Loading cross-encoder...")
        self.cross_encoder_model, self.cross_encoder_tokenizer = self.initialize_cross_encoder()
        print("✅ Cross-encoder loaded")
        
        print(f"🎉 RAG models loaded in {time.time() - start_time:.2f} seconds")

    def initialize_qa_model(self):
        """Initialize QA model pipeline."""
        print("❓ Loading QA model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔧 QA model device: {device}")
        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device
        )

    def initialize_cross_encoder(self):
        """Initialize cross-encoder for reranking."""
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return (
            AutoModelForSequenceClassification.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name)
        )
    

    def simple_tokenize(self, text: str) -> List[str]:
        """Lightweight tokenizer without NLTK."""
        # Remove punctuation and lowercase
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower())
        # Split into words and filter stopwords
        return [word for word in text.split() if word not in STOPWORDS]
    

    def dense_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using vector similarity."""
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                chunk['dense_score'] = float(1/(1 + score))
                results.append(chunk)
        
        return sorted(results, key=lambda x: x['dense_score'], reverse=True)
    

    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using BM25."""
        tokenized_query = self.simple_tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [self.chunks[idx] for idx in top_indices]
    

    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Combine dense and sparse retrieval results."""
        dense_results = self.dense_retrieval(query, top_k)
        sparse_results = self.sparse_retrieval(query, top_k)
        
        results_map = {chunk['id']: chunk for chunk in dense_results}
        for chunk in sparse_results:
            if chunk['id'] not in results_map:
                results_map[chunk['id']] = chunk
        
        for chunk in results_map.values():
            dense_score = chunk.get('dense_score', 0)
            sparse_score = chunk.get('sparse_score', 0)
            chunk['hybrid_score'] = (dense_score + sparse_score) / 2
        
        return sorted(results_map.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]
    

    def generate_answer(self, query: str, max_context_tokens: int = 1024) -> Dict:
        """Generate answer using retrieved context."""
        start_time = time.time()
        
        is_valid, message, category = self.guardrails.validate_query(query)
        if not is_valid:
            return {
                "question": query,
                "answer": message,
                "confidence": 0.95 if category in ["greeting", "out_of_scope"] else 0.97,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Guardrail" if category != "greeting" else "Greeting"
            }
        
        try:
            # Retrieval and processing
            retrieved_chunks = self.hybrid_retrieval(query)
            reranked_chunks = sorted(
                retrieved_chunks,
                key=lambda x: x.get('hybrid_score', 0),
                reverse=True
            )[:3]
            
            # Build context
            context = " ".join(chunk['text'] for chunk in reranked_chunks)
            
            # QA
            result = self.qa_model(question=query, context=context.strip())
            
            confidence = round(min(max(float(result.get('score', 0.0)), 0.0), 1.0), 2)
            
            # Add confidence threshold check
            if confidence < 0.4:
                return {
                    "question": query,
                    "answer": "I'm not confident about this answer. Could you please rephrase your financial question?",
                    "confidence": confidence,
                    "inference_time": round(time.time() - start_time, 4),
                    "method": "Low Confidence"
                }
            
            return {
                "question": query,
                "answer": result.get('answer', "No answer could be generated"),
                "confidence": confidence,
                "inference_time": round(time.time() - start_time, 4),
                "method": "RAG"
            }
        
        except Exception as e:
            return {
                "question": query,
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Error"
            }
