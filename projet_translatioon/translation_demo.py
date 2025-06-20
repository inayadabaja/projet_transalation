import gradio as gr
import ollama  # Remplace OllamaClient
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from utils import format_translation_output
from config import GRADIO_CONFIG, HUGGINGFACE_MODELS, OLLAMA_MODELS, TEMPERATURE_SETTINGS

# Chargement des pipelines HF
hf_pipes = {}
for key, cfg in HUGGINGFACE_MODELS.items():
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    hf_pipes[key] = pipeline(
        "translation",
        model=model,
        tokenizer=tok,
        src_lang="en",
        tgt_lang="fr",
        framework="pt"
    )


def translate(model_key, temp, text):
    if not text.strip():
        return "Please enter some text to translate."
    if model_key.startswith("hf_"):
        model_name = model_key.replace("hf_", "")
        if "t5" in model_name.lower():
            text = f"translate English to French: {text}"
        out = hf_pipes[model_name]([text], max_length=512, truncation=True)
        return out[0]["translation_text"]
    else:
        model_name = model_key.replace("ollama_", "")
        cfg = OLLAMA_MODELS[model_name]
        prompt = TRANSLATION_PROMPTS["en_to_fr"].format(text=text)
        response = ollama.chat(
            model=cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temp}
        )
        return response["message"]["content"]

# Interface Gradio
with gr.Blocks(title=GRADIO_CONFIG["title"], theme=GRADIO_CONFIG["theme"]) as demo:
    with gr.Tabs():
        with gr.Tab("Traduction"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=[f"hf_{k}" for k in HUGGINGFACE_MODELS.keys()] + [f"ollama_{k}" for k in OLLAMA_MODELS.keys()],
                    label="Model"
                )
                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Temperature")

            input_text = gr.Textbox(label="English text to translate", lines=3)
            output_txt = gr.Textbox(label="French translation")
            translate_btn = gr.Button("Translate")

            translate_btn.click(fn=translate, inputs=[model_dropdown, temp_slider, input_text], outputs=output_txt)

        with gr.Tab("📖 Documentation"):
            gr.Markdown("""
                ## 🌍 Projet de Traduction Automatique
                Cette application permet de comparer plusieurs modèles de traduction EN → FR.
                
                ### Modèles disponibles :
                - 🤗 HuggingFace
                - `google-t5/t5-base`
                - `Helsinki-NLP/opus-mt-en-fr`
                - `facebook/mbart-large-50-many-to-many-mmt`
                - 🦙 Ollama
                - `mistral`
                - `llama3`
                - `tinyllama:1.1b`
                - `qwen2:0.5b`
                                
                ### Température
                - **0.0** = réponse déterministe
                - **1.0** = réponse plus créative / aléatoire
                
                ### Fonctionnalités à venir :
                - Traduction FR → EN
                - Évaluation comparative BLEU / ROUGE
                - Téléchargement des résultats
            """)
demo.launch(server_name=GRADIO_CONFIG["server_name"], server_port=GRADIO_CONFIG["server_port"], share=GRADIO_CONFIG["share"])
