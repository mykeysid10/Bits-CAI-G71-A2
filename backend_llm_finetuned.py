"""
Fine-tuned GPT-2 model for financial Q&A with enhanced guardrails.
Combines the working implementation with new features.
"""

import os
import time
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from packaging import version

warnings.filterwarnings("ignore")

class InputGuardrails:
    """Enhanced input validation combining both implementations."""
    
    def __init__(self):
        """Initialize harmful patterns and responses."""
        self.harmful_categories = {
            "violence": {
                "patterns": ["kill", "attack", "shoot", "bomb", "murder"],
                "response": "I cannot assist with violent requests."
            },
            "financial_crime": {
                "patterns": ["launder", "fraud", "scam", "insider trading"],
                "response": "I cannot discuss illegal financial activities."
            },
            "personal_info": {
                "patterns": ["ssn", "credit card", "password", "private key"],
                "response": "I cannot assist with sensitive personal information."
            },
            "out_of_scope": {
                "patterns": ["capital of", "france", "weather", "sports", "movie"],
                "response": "This question is outside my financial expertise."
            },
            "greetings": {
                "patterns": ["hi", "hello", "hey", "how are you", "what's up"],
                "response": "Hello! I specialize in financial questions about Phillips Edison & Company."
            }
        }
    
    def check_input(self, query):
        """Validate input against harmful patterns."""
        query_lower = query.lower().strip()
        for category, data in self.harmful_categories.items():
            if any(pattern in query_lower for pattern in data["patterns"]):
                if category == "greetings":
                    return False, data["response"], "greeting"
                elif category == "out_of_scope":
                    return False, data["response"], "out_of_scope"
                return False, data["response"], "guardrail"
        return True, None, None

class FinancialQAModel:
    """Fine-tuned model combining working implementation with new features."""
    
    def __init__(self, model_path, tokenizer_path):
        """Initialize model with proper device handling."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with proper device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Initialize similarity model with local cache
        os.makedirs('local_models', exist_ok=True)
        self.similarity_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            cache_folder='local_models',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.guardrails = InputGuardrails()
        self.model.eval()  # Set to evaluation mode
    
    def generate_answer(self, question):
        """Generate answer with confidence scoring and input validation."""
        start_time = time.time()
        
        # Input validation
        is_valid, message, category = self.guardrails.check_input(question)
        if not is_valid:
            return {
                "question": question,
                "answer": message,
                "confidence": 0.95 if category in ["greeting", "out_of_scope"] else 0.97,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Input Guardrail"
            }
        
        try:
            # Prepare input with financial-specific prompt
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Configure generation parameters
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 25,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
                "do_sample": True,
                "temperature": 0.001,
                "top_p": 0.001
            }
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Calculate confidence scores
            if hasattr(self.model, 'compute_transition_scores'):
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                avg_confidence = torch.exp(transition_scores[0]).mean().item()
            else:
                # Fallback confidence calculation for older versions
                avg_confidence = 0.7
            
            # Process output
            full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_ans = full_output.split("Answer:")[-1].split("\n")[0].strip()
            
            # Confidence filtering
            if avg_confidence < 0.5:
                return {
                    "question": question,
                    "answer": "I'm not confident about this answer. Please rephrase your financial question.",
                    "confidence": round(avg_confidence, 4),
                    "inference_time": round(time.time() - start_time, 4),
                    "method": "Low Confidence"
                }
            
            return {
                "question": question,
                "answer": generated_ans,
                "confidence": round(avg_confidence, 4),
                "inference_time": round(time.time() - start_time, 4),
                "method": "Fine-tuned GPT-2"
            }
        
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Error"
            }
    
    def __call__(self, question):
        """Alias for generate_answer."""
        return self.generate_answer(question)
