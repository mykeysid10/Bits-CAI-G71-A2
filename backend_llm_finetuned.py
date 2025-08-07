"""
Fine-tuned GPT-2 model for financial Q&A with input guardrails.
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
    """Input validation and guardrails for fine-tuned model."""
    
    def __init__(self):
        """Initialize harmful patterns and responses."""
        self.harmful_categories = {
            "violence": {
                "patterns": ["bomb", "kill", "attack", "shoot", "murder"],
                "response": "I cannot assist with violent or harmful requests."
            },
            "financial_crime": {
                "patterns": ["launder money", "fraud", "insider trading", "scam"],
                "response": "I cannot provide information about illegal financial activities."
            },
            "personal_info": {
                "patterns": ["social security", "credit card", "password", "private key"],
                "response": "I cannot assist with sensitive personal information requests."
            },
            "greetings": {
                "patterns": ["hi", "hello", "hey", "how are you", "what's up"],
                "response": "Hello! I'm a financial Q&A assistant. Please ask me about Phillips Edison & Company's financial statements."
            }
        }
    

    def check_input(self, query):
        """Validate input against harmful patterns.
        
        Args:
            query: User input to validate
            
        Returns:
            Tuple of (is_valid, response, category)
        """
        query_lower = query.lower().strip()
        for category, data in self.harmful_categories.items():
            if any(pattern in query_lower for pattern in data["patterns"]):
                if category == "greetings":
                    return False, data["response"], "greeting"
                return False, data["response"], "guardrail"
        return True, None, None


class FinancialQAModel:
    """Fine-tuned GPT-2 model for financial Q&A."""
    
    def __init__(self, model_path, tokenizer_path):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Initialize model with proper device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        # Move model to device
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        else:
            # For CPU, we don't need to_empty() and can use regular to()
            self.model = self.model.to(self.device)
        
        # self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Create local models directory if it doesn't exist
        os.makedirs('local_models', exist_ok=True)
        # Initialize with local cache
        self.similarity_model = SentenceTransformer(
            'all-MiniLM-L6-v2', 
            cache_folder='local_models',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.guardrails = InputGuardrails()
    

    def generate_answer(self, question):
        """Generate answer to financial question.
        
        Args:
            question: Financial question to answer
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            is_valid, guardrail_response, category_type = self.guardrails.check_input(question)
            if not is_valid:
                if category_type == "greeting":
                    return {
                        "question": question,
                        "answer": guardrail_response,
                        "confidence": 0.95,
                        "inference_time": 0.1,
                        "method": "Greeting"
                    }
                else:
                    return {
                        "question": question,
                        "answer": f"[GUARDRAIL TRIGGERED] {guardrail_response}",
                        "confidence": 0.97,
                        "inference_time": 0.3,
                        "method": "Input Guardrail"
                    }
            
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors = "pt").to(self.device)
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 300,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True
            }
            
            if version.parse(transformers.__version__) >= version.parse("4.0.0"):
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": 0.1,
                    "top_p": 0.9
                })
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits = True
            )
            
            avg_confidence = torch.exp(transition_scores[0]).mean().item()
            full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens = True)
            print(full_output)
            generated_ans = full_output.split("Answer:")[-1].split("\n")[0].strip()
            inference_time = time.time() - start_time
            
            return {
                "question": question,
                "answer": generated_ans,
                "confidence": round(avg_confidence, 4),
                "inference_time": round(inference_time, 4),
                "method": "Fine-tuned GPT-2 Financial QA"
            }
        
        except Exception as e:
            return {
                "question": question,
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "confidence": 0.0,
                "inference_time": 0.0,
                "method": "Error"
            }
