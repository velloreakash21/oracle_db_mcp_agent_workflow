"""
Custom CSS styles for the Streamlit frontend.
"""
import streamlit as st


def apply_custom_styles():
    """Apply custom CSS to the Streamlit app."""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }

    /* Header styling */
    h1 {
        color: #1f4e79;
        font-weight: 700;
    }

    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* Agent activity cards */
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 0.5rem;
    }

    /* Trace visualization */
    .trace-bar {
        font-family: 'Monaco', 'Consolas', monospace;
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        color: #d4d4d4;
    }

    /* Metrics bar */
    .metric-container {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
    }

    /* Input styling */
    .stTextInput input {
        border-radius: 20px;
        padding: 0.75rem 1rem;
    }

    /* Button styling */
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px;
    }

    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
