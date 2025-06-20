"""
Configuration file for the translation demo project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "name": "english_french_translation",
    "source_lang": "en",
    "target_lang": "fr",
    "source_column": "en",
    "target_column": "fr",
    "test_size": 0.1,
    "validation_size": 0.1,
    "sample_size": 1000,  # For demo purposes, use smaller sample
    "random_state": 42
}

# HuggingFace models configuration
HUGGINGFACE_MODELS = {
    "t5_base": {
        "model_name": "t5-base",  # corrigé : nom officiel
        "tokenizer_name": "t5-base",
        "max_length": 512,
        "description": "T5 Base - Text-to-Text Transformer (Google)"
    },
    "marian_en_fr": {
        "model_name": "Helsinki-NLP/opus-mt-en-fr",
        "tokenizer_name": "Helsinki-NLP/opus-mt-en-fr",
        "max_length": 512,
        "description": "MarianMT English → French"
    },
    "marian_fr_en": {
        "model_name": "Helsinki-NLP/opus-mt-fr-en",
        "tokenizer_name": "Helsinki-NLP/opus-mt-fr-en",
        "max_length": 512,
        "description": "MarianMT French → English"
    },
    "mbart50_m2m": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
        "max_length": 512,
        "description": "MBART-50 Multilingual Translation Model"
    }
}

# Ollama models configuration - clés simples, valeurs réelles pour usage
OLLAMA_MODELS = {
    "mistral7b": {
        "model_name": "mistral:7b",
        "source_lang": "en",
        "target_lang": "fr",
        "description": "Ollama Mistral 7B model for English to French translation",
    },
    "llama38b": {
        "model_name": "llama3:8b",
        "source_lang": "en",
        "target_lang": "fr",
        "description": "Ollama Llama 3 8B model for English to French translation",
    },
    "tinyllama1b": {
        "model_name": "tinyllama:1.1b",
        "source_lang": "en",
        "target_lang": "fr",
        "description": "TinyLLaMA 1.1B - modèle léger pour traduction"
    },
    "qwen25b": {
        "model_name": "qwen2:0.5b",
        "source_lang": "en",
        "target_lang": "fr",
        "description": "Qwen 2.5 - Modèle chinois polyvalent pour traduction"
    }
}

# Temperature settings for evaluation
TEMPERATURE_SETTINGS = [0.0, 0.2, 0.5, 0.8, 1.0]

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["bleu", "rouge"],
    "bleu_weights": [
        (1.0,), 
        (0.5, 0.5), 
        (0.33, 0.33, 0.34), 
        (0.25, 0.25, 0.25, 0.25)
    ],
    "rouge_types": ["rouge1", "rouge2", "rougeL"],
    "batch_size": 8
}

# Gradio configuration
GRADIO_CONFIG = {
    "title": "🌍 Machine Translation Demo with LLMs",
    "description": """
    Compare translation quality across different models and temperature settings.
    
    **Models Available:**
    - 🤗 HuggingFace: T5 Base, MarianMT, MBART50
    - 🦙 Ollama: Mistral 7B, Llama 3 8B, TinyLLaMA 1.1B, Qwen 2.5
    
    **Languages:** English ↔ French
    """,
    "theme":  {
        "primary_hue": "#5A20CB",        # violet foncé
        "secondary_hue": "#FF61A6",      # rose vif
        "background_fill": "#FAFAFC",    # très clair
        "text_primary": "#2E2E3A",       # gris foncé
        "text_secondary": "#6B6B83",     # gris moyen
        "font": "Poppins, sans-serif",   # police moderne et douce
        "radius_size": "12px"             # arrondis plus marqués
        },
    "share": False,
    "server_port": 7860,
    "server_name": "127.0.0.1"
}

# Translation prompts for LLM models
TRANSLATION_PROMPTS = {
    "en_to_fr": "Translate the following English text to French: '{text}'",
    "fr_to_en": "Translate the following French text to English: '{text}'",
    "system_prompt": (
        "You are a professional translator. "
        "Provide only the translation without any additional text or explanation."
    )
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": PROJECT_ROOT / "translation_demo.log"
}

# Performance settings
PERFORMANCE_CONFIG = {
    "max_workers": 4,
    "timeout": 30,
    "retry_attempts": 3,
    "cache_enabled": True
}
