"""
Fine-tuned GPT-2 model for financial Q&A with enhanced guardrails and complete sentence generation.
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
                "patterns": ["launder", "fraud", "scam"],
                "response": "I cannot discuss illegal financial activities."
            },
            "out_of_scope": {
                "patterns": ["capital of", "france", "weather", "sports"],
                "response": "This question is outside my financial expertise."
            },
            "greetings": {
                "patterns": ["hi", "hello", "hey"],
                "response": "Hello! Please ask about financial statements."
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
    """Fine-tuned model with confidence filtering and complete sentence generation."""
    
    def __init__(self, model_path, tokenizer_path):
        """Initialize model components."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.guardrails = InputGuardrails()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate_answer(self, question):
        """Generate answer with validation, confidence checks, and strict cleanup."""
        start_time = time.time()
        
        # Input validation
        is_valid, message, category = self.guardrails.check_input(question)
        if not is_valid:
            return {
                "question": question,
                "answer": message,
                "confidence": 0.95 if category in ["greeting", "out_of_scope"] else 0.97,
                "inference_time": round(time.time() - start_time, 4),
                "method": category.replace("_", " ").title()
            }
        
        try:
            # Prepare input with financial-specific prompt
            prompt = f"Question: {question}\nFinancial Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response with settings optimized for complete sentences
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.001,
                top_p=0.001,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Calculate confidence scores
            token_probs = []
            for step_scores, token_id in zip(outputs.scores, outputs.sequences[0][inputs.input_ids.shape[1]:]):
                step_probs = step_scores.softmax(dim=-1)
                token_probs.append(step_probs[0, token_id].item())

            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.1
            
            # Decode and process output
            full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            answer_raw = full_output.split("Financial Answer:")[-1].strip()
            
            # Enhanced sentence processing from working local version
            # Step 1: Split into sentences (handling ". " delimiter)
            sentences = [s.strip() for s in answer_raw.split('. ') if s.strip()]
            
            # Step 2: Remove duplicates while preserving order
            seen = set()
            dedup_sentences = []
            for sent in sentences:
                if sent not in seen:
                    seen.add(sent)
                    dedup_sentences.append(sent)
            
            # Step 3: Reconstruct answer with proper punctuation
            clean_answer = '. '.join(dedup_sentences) + '.' if dedup_sentences else answer_raw
            
            # Fallback if no sentences found (e.g., if model didn't use periods)
            if not clean_answer.strip('.').strip():
                clean_answer = answer_raw.split('\n')[0][:100]  # Take first line or first 100 chars
            
            # Confidence filtering
            if confidence < 0.5:
                return {
                    "question": question,
                    "answer": "I'm not confident about this answer. Please rephrase your financial question.",
                    "confidence": round(confidence, 4),
                    "inference_time": round(time.time() - start_time, 4),
                    "method": "Low Confidence"
                }
            
            return {
                "question": question,
                "answer": clean_answer,
                "confidence": round(confidence, 4),
                "inference_time": round(time.time() - start_time, 4),
                "method": "Fine-tuned"
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
