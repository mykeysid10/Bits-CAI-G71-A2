# 💰 Financial Q&A System for Phillips Edison & Company.

Developed as part of BITS CAI Assignment II by Group 71 [Sarit, Dhiman, Soumen, Omkar, Siddharth]

---

## 🌟 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bits-cai-group-71-assignment-2.streamlit.app/)

## 📌 Overview

A dual-approach question answering system designed specifically for financial queries about Phillips Edison & Company's financial statements. 
Implements both:
1. **Retrieval-Augmented Generation (RAG)**
2. **Fine-tuned GPT-2 Model**

## 📊 Core Dataset: `financial_qna_pairs.csv`

## ✨ Key Features

- **Input guardrails** to filter harmful/inappropriate queries.
- **Hybrid retrieval** combining BM25 and FAISS vector search.
- **Cross-encoder reranking** for improved relevance.
- **Domain Knowledge** using GPT-2-medium Finetuning.
- **Streamlit-based GUI** for intuitive interaction.

## 🛠️ Technical Stack

| Component               | Technology Used                          |
|-------------------------|------------------------------------------|
| Language Model          | Fine-tuned GPT-2-medium                  |
| Vector Database         | FAISS                                    |
| Sparse Retrieval        | BM25                                     |
| Reranking               | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| Embeddings              | all-MiniLM-L6-v2                         |
| QA Pipeline             | roberta-base-squad2                      |
| Frontend                | Streamlit                                |

## 🎬 Demo

[https://github.com/mykeysid10/Bits-CAI-G71-A2/blob/main/Hosted_Demo_Recording.mp4](https://github.com/user-attachments/assets/318c41d7-6208-424b-a6cb-167a83a94750)
