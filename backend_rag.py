"""
Retrieval-Augmented Generation (RAG) system for financial Q&A.
Combines dense and sparse retrieval with cross-encoder reranking.
"""

import os
import torch
import time
import pickle
import faiss
import string
import warnings
from typing import Dict, Tuple, List
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class RAGGuardrails:
    """Input validation and guardrails for RAG system."""
    
    def __init__(self):
        """Initialize harmful patterns and responses."""
        self.harmful_patterns = {
            "violence": ["kill", "attack", "shoot", "bomb"],
            "financial_crime": ["launder", "fraud", "scam"],
            "personal_info": ["ssn", "credit card", "password"],
            "greetings": ["hi", "hello", "hey", "how are you", "what's up"]
        }
    

    def validate_query(self, query: str) -> Tuple[bool, str, str]:
        """Check for harmful or greeting queries.
        
        Args:
            query: User input query to validate
            
        Returns:
            Tuple of (is_valid, message, category)
        """
        query_lower = query.lower().strip()
        
        if any(pattern in query_lower for pattern in self.harmful_patterns["greetings"]):
            return False, "Hello! I'm a financial Q&A assistant. Please ask me about financial statements.", "greeting"
        
        for category, patterns in self.harmful_patterns.items():
            if category == "greetings":
                continue
            if any(pattern in query_lower for pattern in patterns):
                return False, f"I cannot answer questions related to {category.replace('_', ' ')}.", "guardrail"
        
        if len(query_lower.split()) < 3:
            return False, "Please provide a more detailed question.", "short_query"
            
        return True, "", "valid"


class RAGSystem:
    """Main RAG system implementation."""
    
    def __init__(self, artifacts_dir = "rag-artifacts"):
        """Initialize RAG system with pre-built indexes.
        
        Args:
            artifacts_dir: Directory containing pre-built indexes and artifacts
        """
        self.artifacts_dir = artifacts_dir
        self.guardrails = RAGGuardrails()
        self.load_models()
        self.qa_model = self.initialize_qa_model()
    

    def load_models(self):
        """Load pre-built models and indexes."""
        print("Loading RAG models and indexes...")
        start_time = time.time()
        
        # self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Create local models directory if it doesn't exist
        os.makedirs('local_models', exist_ok = True)
        # Initialize with local cache
        self.embed_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            cache_folder = 'local_models',
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.faiss_index = faiss.read_index(os.path.join(self.artifacts_dir, "faiss_index.index"))
        
        with open(os.path.join(self.artifacts_dir, "bm25_index.pkl"), "rb") as f:
            self.bm25_index, self.chunks = pickle.load(f)
        
        self.cross_encoder_model, self.cross_encoder_tokenizer = self.initialize_cross_encoder()
        
        print(f"Models loaded in {time.time() - start_time:.2f} seconds")
    

    def initialize_qa_model(self):
        """Initialize QA model pipeline.
        
        Returns:
            Initialized QA pipeline
        """
        return pipeline(
            "question-answering",
            model = "deepset/roberta-base-squad2",
            device = 0 if torch.cuda.is_available() else -1
        )
    

    def initialize_cross_encoder(self):
        """Initialize cross-encoder for reranking.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return (
            AutoModelForSequenceClassification.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name)
        )
    

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query for BM25 search.
        
        Args:
            query: Input query string
            
        Returns:
            List of preprocessed tokens
        """
        query = query.lower().translate(str.maketrans('', '', string.punctuation))
        return [word for word in word_tokenize(query) if word not in stopwords.words("english")]
    

    def dense_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using vector similarity.
        
        Args:
            query: Input query
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        query_embedding = self.embed_model.encode([query], convert_to_numpy = True)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                chunk['dense_score'] = float(1/(1 + score))  # Convert distance to similarity
                results.append(chunk)
        
        return sorted(results, key = lambda x: x['dense_score'], reverse = True)
    

    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using BM25.
        
        Args:
            query: Input query
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        tokenized_query = self.preprocess_query(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk['sparse_score'] = float(scores[idx])
            results.append(chunk)
            
        return results
    

    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """Combine dense and sparse retrieval results.
        
        Args:
            query: Input query
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with hybrid scores
        """
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
        
        return sorted(results_map.values(), key = lambda x: x['hybrid_score'], reverse = True)[:top_k]
    

    def rerank_with_cross_encoder(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank retrieved chunks using cross-encoder.
        
        Args:
            query: Original query
            chunks: Retrieved chunks to rerank
            top_k: Number of results to return
            
        Returns:
            List of reranked chunks with scores
        """
        features = self.cross_encoder_tokenizer(
            [query] * len(chunks),
            [chunk['text'] for chunk in chunks],
            padding = True,
            truncation = True,
            return_tensors = "pt"
        )
        
        with torch.no_grad():
            scores = self.cross_encoder_model(**features).logits.squeeze()
        
        for chunk, score in zip(chunks, scores):
            chunk['cross_encoder_score'] = float(score)
            
        return sorted(chunks, key = lambda x: x['cross_encoder_score'], reverse = True)[:top_k]
    

    def generate_answer(self, query: str, max_context_tokens: int = 1024) -> Dict:
        """Generate answer using retrieved context.
        
        Args:
            query: User question
            max_context_tokens: Maximum context tokens for QA
            
        Returns:
            Dictionary containing answer and metadata
        """
        start_time = time.time()
        
        is_valid, message, category = self.guardrails.validate_query(query)
        if not is_valid:
            return {
                "question": query,
                "answer": message,
                "confidence": 0.95 if category == "greeting" else 0.97,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Guardrail" if category != "greeting" else "Greeting"
            }
        
        try:
            retrieval_start = time.time()
            retrieved_chunks = self.hybrid_retrieval(query)
            retrieval_time = time.time() - retrieval_start
            
            reranking_start = time.time()
            reranked_chunks = self.rerank_with_cross_encoder(query, retrieved_chunks)
            reranking_time = time.time() - reranking_start
            
            context = ""
            token_count = 0
            sources = set()
            
            for chunk in reranked_chunks:
                chunk_tokens = word_tokenize(chunk['text'])
                if token_count + len(chunk_tokens) > max_context_tokens:
                    break
                context += chunk['text'] + " "
                token_count += len(chunk_tokens)
                if 'source' in chunk:
                    sources.add(chunk['source'])
            
            qa_start = time.time()
            result = self.qa_model(question = query, context = context.strip())
            qa_time = time.time() - qa_start
            
            confidence = result.get('score', 0.0)
            if 'confidence' in result:
                confidence = result['confidence']
            
            confidence = min(max(float(confidence), 0.0), 1.0)
            
            return {
                "question": query,
                "answer": result.get('answer', "No answer could be generated"),
                "confidence": round(confidence, 4),
                "inference_time": round(time.time() - start_time, 4),
                "method": "RAG",
                "details": {
                    "retrieval_time": round(retrieval_time, 4),
                    "reranking_time": round(reranking_time, 4),
                    "qa_time": round(qa_time, 4),
                    "sources": list(sources) if sources else [],
                    "context_used": context.strip()
                }
            }
    
        except Exception as e:
            return {
                "question": query,
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "confidence": 0.0,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Error"
            }
    

    def __call__(self, query: str) -> Dict:
        """Alias for generate_answer.
        
        Args:
            query: User question
            
        Returns:
            Dictionary containing answer and metadata
        """
        return self.generate_answer(query)
