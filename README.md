# 💰 Financial Q&A System for Phillips Edison & Company

Developed as part of BITS CAI Assignment II by Group 71 [Sarit, Dhiman, Soumen, Omkar, Siddharth]

---

## 🌟 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bits-cai-group-71-assignment-2.streamlit.app/)

## 📌 Overview

A dual-approach question answering system designed specifically for financial queries about Phillips Edison & Company's financial statements. 
Implements both:
1. **Retrieval-Augmented Generation (RAG)**
2. **Fine-tuned GPT-2 Model**

## 📊 Core Dataset: `financial_qa_sample_data.csv`

**Purpose**:  This dataset serves as the foundation for both:
- Fine-tuning the GPT-2 model
- Building the RAG retrieval index

## ✨ Key Features

- **Input guardrails** to filter harmful/inappropriate queries.
- **Hybrid retrieval** combining BM25 and FAISS vector search.
- **Cross-encoder reranking** for improved relevance.
- **Streamlit-based GUI** for intuitive interaction.

## 🛠️ Technical Stack

| Component               | Technology Used                          |
|-------------------------|------------------------------------------|
| Language Model          | Fine-tuned GPT-2                         |
| Vector Database         | FAISS                                    |
| Sparse Retrieval        | BM25                                     |
| Reranking               | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| Embeddings              | all-MiniLM-L6-v2                         |
| QA Pipeline             | roberta-base-squad2                      |
| Frontend                | Streamlit                                |
