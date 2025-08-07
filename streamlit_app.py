"""
Streamlit-based GUI for Financial Q&A System.
Handles user interactions and displays responses from both RAG and fine-tuned models.
"""

import streamlit as st
import pandas as pd
from backend_llm_finetuned import FinancialQAModel
from backend_rag import RAGSystem

# Sample data - this would be your 75 Q&A pairs
SAMPLE_DATA = pd.read_csv("financial_qna_pairs.csv")

def load_css():
    """Load custom CSS styles for the application."""
    st.markdown("""
    <style>
        .main {
            background-color: #1E3F66;
        }
        .sidebar .sidebar-content {
            background-color: #0A1F3D;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFA500;
        }
        .stTextInput>div>div>input {
            background-color: #F0F2F6;
            color: #000000;
        }
        .stButton>button {
            background-color: #FFA500 !important;
            color: white !important;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FFA500 !important;
            color: white !important;
        }
        .chat-message {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .user-message {
            background-color: #E1F5FE;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #F5F5F5;
            margin-right: 20%;
        }
        .info-text {
            font-size: 0.8em;
            color: #666;
        }
        .chat-input-container {
            position: fixed;
            bottom: 20px;
            width: 83%;
            background-color: #1E3F66;
            padding: 10px;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .exit-btn {
            background-color: #f44336 !important;
            color: white !important;
            border-radius: 5px;
            border: none;
        }
        .exit-btn:hover {
            background-color: #d32f2f !important;
            color: white !important;
        }
        .sample-data-table {
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }
        .sample-data-table th, .sample-data-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .sample-data-table th {
            background-color: #FFA500;
            color: white;
        }
        .sample-data-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "ft_chat_history": [],
        "rag_chat_history": [],
        "current_mode": "Home",
        "ft_model": None,
        "rag_model": None,
        "models_loaded": False,
        "input_key": 0,
        "exit_clicked": False,
        "processing": False,
        "show_typing": False,
        "show_sample_data": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def loading_screen():
    """Display loading screen while models are being loaded."""
    st.markdown("<h1 style='text-align: center;'>Loading Financial Q&A System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align: center;'><img src='https://i.gifer.com/ZZ5H.gif' width='200'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please wait while we load the models...</p>", unsafe_allow_html=True)


def load_models():
    """Load RAG and fine-tuned models."""
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            try:
                st.session_state.rag_model = RAGSystem(artifacts_dir="rag-artifacts")
                st.session_state.ft_model = FinancialQAModel(
                    "finetuned-gpt2-artifacts", 
                    "finetuned-gpt2-artifacts"
                )
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()


def home_page():
    """Display home page with system information."""
    st.markdown("<h1 style='text-align: center;'>PHILLIPS EDISON & COMPANY, INC.</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #FFA500;'>Financial Q&A System</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>BITS CAI Assignment II</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Group 71: Sarit Ghosh, Dhiman Kundu, Soumen Choudhury, Omkar Patil, Siddharth Kulkarni</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### About This System
    This application provides two approaches for answering questions about our company's financial statements:
    1. **Retrieval-Augmented Generation**: Uses Retrieval (FAISS + BM25 Search), NLG (Roberta QnA), Reranking (Cross Encoder: ms-marco-MiniLM-L-6-v2).
    2. **Fine-Tuned Model**: Uses a language model (GPT-2) fine-tuned on financial Q&A pairs.

    **Note:** Due to a 1GB space constraint in the deployed environment, we're using a fine-tuned GPT-2 model instead of the larger GPT-2-medium model. However, all reported results in our documentation are based on GPT-2-medium.
    
    **Note:** The Finetuned-LLM tab may respond slowly (~ 2 to 3 min per query).
    
    Select an option from the sidebar to test either system!
    """)


def chat_interface(mode):
    """Display chat interface for the selected mode."""
    st.session_state.current_mode = mode
    chat_history = st.session_state.ft_chat_history if mode == "Fine-Tuned" else st.session_state.rag_chat_history
    model = st.session_state.ft_model if mode == "Fine-Tuned" else st.session_state.rag_model
    
    st.markdown(f"<h2 style='text-align: center;'>{mode} Q&A System</h2>", unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <span class="emoji">👤</span>
                    <strong>User:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <span class="emoji">🤖</span>
                    <strong>Assistant:</strong> {msg["content"]}<br>
                    <span class="info-text">
                        Confidence: {msg.get('confidence', 'N/A')} | 
                        Time: {msg.get('time', 'N/A')}s | 
                        Method: {msg.get('method', 'N/A')}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.show_typing:
            st.markdown("""
            <div class="chat-message bot-message">
                <span class="emoji">🤖</span>
                <em>Assistant is typing...</em>
            </div>
            """, unsafe_allow_html=True)

    with st.form(key = f"{mode}_chat_form", clear_on_submit = True):
        user_input = st.text_input(
            "", 
            key = f"input_{st.session_state.input_key}", 
            placeholder = "Type your question here...", 
            label_visibility = "collapsed"
        )
        
        submit_button = st.form_submit_button("Submit")
        exit_button = st.form_submit_button("Exit", type = "primary")
        
        if submit_button and user_input.strip():
            chat_history.append({"role": "user", "content": user_input})
            st.session_state.input_key += 1
            st.session_state.processing = True
            st.session_state.show_typing = True
            st.rerun()
        
        if exit_button:
            st.session_state.exit_clicked = True
            st.rerun()

    if chat_history and chat_history[-1]["role"] == "user" and st.session_state.processing:
        with st.spinner("Processing..."):
            try:
                question = chat_history[-1]["content"]
                response = model.generate_answer(question)
                chat_history.append({
                    "role": "bot",
                    "content": response["answer"],
                    "confidence": response["confidence"],
                    "time": response["inference_time"],
                    "method": response["method"]
                })
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                chat_history.append({
                    "role": "bot",
                    "content": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "time": 0.0,
                    "method": "Error"
                })
            finally:
                st.session_state.processing = False
                st.session_state.show_typing = False
                st.rerun()


def exit_options(mode):
    """Display exit options after chat ends."""
    st.markdown(f"<h3>{mode} Chat Ended</h3>", unsafe_allow_html=True)
    chat_history = st.session_state.ft_chat_history if mode == "Fine-Tuned" else st.session_state.rag_chat_history
    history_text = "\n\n".join(
        f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}\n(Confidence: {msg.get('confidence', 'N/A')}, Time: {msg.get('time', 'N/A')}s)"
        for msg in chat_history
    )
    st.download_button("Download Chat History", history_text, file_name = f"{mode}_chat_history.txt")
    
    if st.button("Start New Chat"):
        if mode == "Fine-Tuned":
            st.session_state.ft_chat_history = []
        else:
            st.session_state.rag_chat_history = []
        st.session_state.exit_clicked = False
        st.session_state.input_key += 1
        st.rerun()


def show_sample_data():
    """Display the sample Q&A data."""
    st.markdown("<h2 style='text-align: center;'>Sample Q&A Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is the dataset used for RAG & Fine-tuning the model</p>", unsafe_allow_html=True)
    
    # Display the sample data
    st.dataframe(SAMPLE_DATA, use_container_width=True)
    
    # Add download button for the sample data
    csv = SAMPLE_DATA.to_csv(index=False)
    st.download_button(
        label="Download Sample Data as CSV",
        data=csv,
        file_name='financial_qa_sample_data.csv',
        mime='text/csv'
    )
    

def main():
    """Main application function."""
    st.set_page_config(
        page_title = "Financial Q&A System",
        page_icon = ":chart_with_upwards_trend:",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )
    
    load_css()
    init_session_state()
    
    if not st.session_state.models_loaded:
        loading_screen()
        load_models()
        st.rerun()
    
    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio("Go to", ["Home", "RAG", "Fine-Tuned LLM", "Sample Data"])
    
    mode_mapping = {
        "Home": "Home",
        "RAG": "RAG",
        "Fine-Tuned LLM": "Fine-Tuned",
        "Sample Data": "Sample Data"
    }
    
    selected_mode = mode_mapping[nav_option]
    
    if selected_mode != st.session_state.current_mode:
        if selected_mode == "Sample Data":
            st.session_state.show_sample_data = True
        else:
            st.session_state.ft_chat_history = []
            st.session_state.rag_chat_history = []
            st.session_state.exit_clicked = False
            st.session_state.current_mode = selected_mode
            st.session_state.show_sample_data = False
        st.session_state.input_key += 1
    
    if st.session_state.show_sample_data:
        show_sample_data()
    elif st.session_state.current_mode == "Home":
        home_page()
    elif st.session_state.current_mode in ["RAG", "Fine-Tuned"]:
        if not st.session_state.exit_clicked:
            chat_interface(st.session_state.current_mode)
        else:
            exit_options(st.session_state.current_mode)


if __name__ == "__main__":
    main()
