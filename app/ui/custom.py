"""
Defines custom CSS for the application.
"""
import streamlit as st

def load_custom_css():
    """
    Load custom CSS to enhance the application's UI.
    """
    custom_css = """
    <style>
        /* Main app styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Card-like styling for containers */
        div[data-testid="stVerticalBlock"] > div.element-container:not(:first-child) {
            background-color: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin: 0.5rem 0;
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        }
        
        /* Metric styles */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        }
        
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Footer styling */
        .app-footer {
            margin-top: 3rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #333;
        }
        
        /* Search box styling */
        input[type="text"] {
            border-radius: 0.5rem;
            border-color: #DDD;
        }
        
        /* Add animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .main .block-container {
            animation: fadeIn 0.5s ease-in-out;
        }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def styled_header(text, level=1):
    """
    Create a styled header with custom formatting.
    
    Args:
        text: Header text
        level: Header level (1-3)
    """
    if level == 1:
        st.markdown(f"<h1 style='margin-bottom: 1rem; color: #333; font-size: 2.2rem;'>{text}</h1>", 
                    unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f"<h2 style='margin-bottom: 0.8rem; color: #444; font-size: 1.8rem;'>{text}</h2>", 
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='margin-bottom: 0.6rem; color: #555; font-size: 1.5rem;'>{text}</h3>", 
                    unsafe_allow_html=True) 