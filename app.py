import streamlit as st
from langdetect import detect
from transformers import pipeline
from utils.translation import translate_text
from utils.summarization import summarize_chat

# App configuration
st.set_page_config(page_title="Multilingual Chat Summarizer", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    model_size = st.radio(
        "Model Size",
        ["Small (faster)", "Large (more accurate)"],
        index=0
    )
    target_language = st.selectbox(
        "Summary Language",
        ["English", "Spanish", "French", "German", "Chinese", "Same as input"],
        index=0
    )

# Main interface
st.title("Multilingual Chat Summarizer")
chat_input = st.text_area("Paste your chat conversation here:", height=300)

if st.button("Generate Summary"):
    if not chat_input.strip():
        st.warning("Please enter some chat text to summarize")
    else:
        with st.spinner("Analyzing conversation..."):
            # Detect language if needed
            if target_language == "Same as input":
                try:
                    lang = detect(chat_input[:500])  # Sample first 500 chars for speed
                    target_language = {
                        'en': 'English',
                        'es': 'Spanish',
                        'fr': 'French',
                        'de': 'German',
                        'zh': 'Chinese'
                    }.get(lang, 'English')
                except:
                    target_language = 'English'
            
            # Generate summary
            summary = summarize_chat(chat_input, model_size, target_language)
            
            # Display results
            st.subheader(f"Summary ({target_language})")
            st.write(summary)
            
            # Show stats
            orig_words = len(chat_input.split())
            summ_words = len(summary.split())
            reduction = int(100 * (1 - summ_words/orig_words))
            st.caption(f"Reduced from {orig_words} to {summ_words} words ({reduction}% reduction)")