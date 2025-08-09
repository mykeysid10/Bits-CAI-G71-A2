"""
Fine-tuned GPT-2 model for financial Q&A with enhanced guardrails.
Optimized for deployment with concise answers and better input validation.
"""

import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class InputGuardrails:
    """Enhanced input validation for fine-tuned model."""
    
    def __init__(self):
        """Initialize guardrail patterns and responses."""
        self.harmful_categories = {
            "violence": {
                "patterns": ["kill", "attack", "shoot", "bomb"],
                "response": "I cannot assist with violent requests."
            },
            "financial_crime": {
                "patterns": ["launder", "fraud", "scam", "insider trading"],
                "response": "I cannot discuss illegal financial activities."
            },
            "personal_info": {
                "patterns": ["social security", "credit card", "password"],
                "response": "I cannot assist with sensitive personal information."
            },
            "out_of_scope": {
                "patterns": ["capital of", "weather", "sports", "movie"],
                "response": "This question is outside my financial expertise."
            },
            "greetings": {
                "patterns": ["hi", "hello", "hey", "how are you"],
                "response": "Hello! I specialize in financial questions about Phillips Edison & Company."
            }
        }
    
    def check_input(self, query):
        """Validate input against harmful patterns."""
        query_lower = query.lower().strip()
        for category, data in self.harmful_categories.items():
            if any(pattern in query_lower for pattern in data["patterns"]):
                return False, data["response"], category
        return True, None, None

class FinancialQAModel:
    """Fine-tuned model with confidence filtering and concise answers."""
    
    def __init__(self, model_path, tokenizer_path):
        """Initialize model components with deployment constraints."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with proper device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with appropriate settings for deployment
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.guardrails = InputGuardrails()
    
    def generate_answer(self, question):
        """
        Generate financial answer with:
        - Input validation
        - Confidence scoring
        - Strict answer cleanup
        - Controlled response length
        """
        start_time = time.time()
        
        # Input validation
        is_valid, message, category = self.guardrails.check_input(question)
        if not is_valid:
            return {
                "question": question,
                "answer": message,
                "confidence": 0.95,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Input Guardrail"
            }
        
        try:
            # Prepare input with financial-specific prompt
            inputs = self.tokenizer(
                f"Question: {question}\nFinancial Answer:",
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response with conservative settings
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=75,  # Increased for better answers but still controlled
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Calculate confidence from token probabilities
            token_probs = []
            for step_scores, token_id in zip(outputs.scores, outputs.sequences[0][inputs.input_ids.shape[1]:]):
                step_probs = step_scores.softmax(dim=-1)
                token_probs.append(step_probs[0, token_id].item())
            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.1
            
            # Decode and clean output
            full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            answer_raw = full_output.split("Financial Answer:")[-1].strip()
            
            # Extract first complete answer and remove duplicates
            sentences = re.split(r'(?<!\d)\.(?!\d)\s*', answer_raw)
            sentences = [s.strip() for s in sentences if s.strip()]
            clean_answer = sentences[0] + '.' if sentences else answer_raw
            
            # Remove any repeated phrases
            clean_answer = re.sub(r'\b(\w+)\b(?=.*\b\1\b)', '', clean_answer, flags=re.I)
            clean_answer = ' '.join(clean_answer.split())  # Normalize whitespace
            
            # Confidence filter
            if confidence < 0.5:
                return {
                    "question": question,
                    "answer": "I'm not confident about this answer. Please ask a different financial question.",
                    "confidence": round(confidence, 4),
                    "inference_time": round(time.time() - start_time, 4),
                    "method": "Low Confidence"
                }
            
            return {
                "question": question,
                "answer": clean_answer[:500],  # Hard limit for safety
                "confidence": round(confidence, 4),
                "inference_time": round(time.time() - start_time, 4),
                "method": "Fine-tuned GPT-2"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing your question: {str(e)}",
                "confidence": 0.0,
                "inference_time": round(time.time() - start_time, 4),
                "method": "Error"
            }
