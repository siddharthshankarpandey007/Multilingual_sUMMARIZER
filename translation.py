from transformers import pipeline
from langdetect import detect, LangDetectException
from functools import lru_cache
import torch

# Language to model mapping (using Helsinki-NLP OPUS-MT models)
LANGUAGE_MODELS = {
    "Spanish": {
        "code": "es",
        "to_english": "Helsinki-NLP/opus-mt-es-en",
        "from_english": "Helsinki-NLP/opus-mt-en-es"
    },
    "French": {
        "code": "fr",
        "to_english": "Helsinki-NLP/opus-mt-fr-en",
        "from_english": "Helsinki-NLP/opus-mt-en-fr"
    },
    "German": {
        "code": "de",
        "to_english": "Helsinki-NLP/opus-mt-de-en",
        "from_english": "Helsinki-NLP/opus-mt-en-de"
    },
    "Chinese": {
        "code": "zh",
        "to_english": "Helsinki-NLP/opus-mt-zh-en",
        "from_english": "Helsinki-NLP/opus-mt-en-zh"
    }
}

@lru_cache(maxsize=2)
def get_translator(model_name):
    """Cache translator instances to avoid reloading models"""
    return pipeline(
        "translation",
        model=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def translate_text(text, target_language):
    if not text.strip():
        return text
    
    if target_language == "English":
        return text
    
    target_info = LANGUAGE_MODELS.get(target_language)
    if not target_info:
        print(f"Unsupported target language: {target_language}")
        return text
    
    try:
        # Detect source language with fallback
        try:
            src_lang = detect(text[:500])  # More efficient than 1000 chars
        except LangDetectException:
            src_lang = "en"
        
        # Direct translation path if available
        if src_lang == target_info["code"]:
            return text
            
        if src_lang != "en":
            # Two-step translation: src -> English -> target
            to_english = get_translator(target_info["to_english"])
            english_text = to_english(text)[0]['translation_text']
            
            from_english = get_translator(target_info["from_english"])
            translated_text = from_english(english_text)[0]['translation_text']
            return translated_text
        else:
            # Direct translation from English
            translator = get_translator(target_info["from_english"])
            return translator(text)[0]['translation_text']
            
    except Exception as e:
        print(f"Translation error: {e}")
        return text

