from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from .translation import translate_text

nltk.download('punkt')

def summarize_chat(text, model_size="Small (faster)", target_language="English"):
    # Select model based on size preference
    model_name = "facebook/bart-large-cnn" if model_size == "Large (more accurate)" else "sshleifer/distilbart-cnn-12-6"
    
    # Initialize summarization pipeline
    summarizer = pipeline("summarization", model=model_name)
    
    # Handle long texts by chunking
    max_chunk_size = 1024 if model_size == "Large (more accurate)" else 512
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except:
            continue
    
    # Combine chunk summaries
    combined_summary = ' '.join(summaries)
    
    # Translate if needed
    if target_language != "English":
        combined_summary = translate_text(combined_summary, target_language)
    
    # Post-process for better readability
    sentences = sent_tokenize(combined_summary)
    return ' '.join(sentences[:5])