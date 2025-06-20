import gradio as gr
import ollama
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from utils import format_translation_output
from config import GRADIO_CONFIG, HUGGINGFACE_MODELS, OLLAMA_MODELS, TEMPERATURE_SETTINGS, TRANSLATION_PROMPTS

from visualiser import ResultsVisualizer  # Tu peux mettre ta classe dans un fichier `visualizer_module.py`

from pathlib import Path
import logging

# --- Initialisation des pipelines HF ---
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

def translate(direction, model_key, temp, text):
    if not text.strip():
        return "Please enter some text to translate."

    is_hf = model_key.startswith("hf_")
    model_name = model_key.replace("hf_", "").replace("ollama_", "")

    if is_hf:
        # HF n’a pas toujours besoin du src_lang/tgt_lang si le modèle est spécifique
        if "t5" in model_name.lower():
            prefix = "translate English to French" if direction == "en_to_fr" else "translate French to English"
            text = f"{prefix}: {text}"
        out = hf_pipes[model_name]([text], max_length=512, truncation=True)
        return out[0]["translation_text"]
    else:
        cfg = OLLAMA_MODELS[model_name]
        prompt = TRANSLATION_PROMPTS[direction].format(text=text)
        try:
            response = ollama.chat(
                model=cfg["model_name"],
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Ollama error: {e}"

# --- Initialisation du visualizer ---
results_dir_path = Path("results")
visualizer = ResultsVisualizer()
visualizer.load_results(results_dir_path)

available_models = visualizer.get_available_models()
available_metrics = visualizer.get_available_metrics()
available_temperatures = visualizer.get_available_temperatures()

default_model = available_models[0] if available_models else None
default_metric = available_metrics[0] if available_metrics else None
default_temp = available_temperatures[0] if available_temperatures else None

# --- Gradio Interface combinée ---
# --- Gradio Interface combinée ---
with gr.Blocks(title="Traduction & Visualisation", theme=GRADIO_CONFIG["theme"]) as demo:

    # Injection CSS pour personnaliser boutons, ombres, etc.
    gr.HTML("""
    <style>
        /* Police principale */
        body, .gradio-container {
            font-family: 'Poppins', sans-serif !important;
            background-color: #FAFAFC !important;
            color: #2E2E3A !important;
        }

        /* Boutons */
        .gr-button {
            background: linear-gradient(135deg, #5A20CB 0%, #FF61A6 100%);
            border: none;
            color: white !important;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(90, 32, 203, 0.3);
            border-radius: 12px !important;
            transition: background 0.3s ease;
        }
        .gr-button:hover {
            background: linear-gradient(135deg, #FF61A6 0%, #5A20CB 100%);
            box-shadow: 0 6px 14px rgba(255, 97, 166, 0.4);
        }

        /* Onglets */
        .gr-tabs-header {
            font-weight: 700;
            font-size: 1.1rem;
            color: #5A20CB !important;
            border-bottom: 3px solid #FF61A6;
            padding-bottom: 8px;
        }

        /* Conteneurs */
        .gr-row, .gr-column {
            padding: 12px;
        }

        /* Inputs (dropdown, sliders, etc.) */
        .gr-input, .gr-dropdown, .gr-slider {
            border-radius: 12px !important;
            border: 1.5px solid #CCC;
            padding: 8px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
            transition: border-color 0.3s ease;
        }
        .gr-input:focus, .gr-dropdown:focus, .gr-slider:focus {
            border-color: #5A20CB !important;
            box-shadow: 0 0 8px rgba(90, 32, 203, 0.4);
        }
    </style>
    """)

    # Ajout d'un titre et d'une description avant les onglets
    gr.Markdown("# Bienvenue dans l'application de traduction")
    gr.Markdown("Cette application vous permet de traduire du texte et de visualiser les résultats selon différents modèles et paramètres.")

    with gr.Tabs():
        # 🟢 Onglet Traduction
        with gr.Tab("🔄 Traduction"):
            with gr.Row():
                direction_dropdown = gr.Dropdown(
                    choices=["en_to_fr", "fr_to_en"],
                    label="Direction",
                    value="en_to_fr"
                )
                model_dropdown = gr.Dropdown(
                    choices=[f"hf_{k}" for k in HUGGINGFACE_MODELS.keys()] + [f"ollama_{k}" for k in OLLAMA_MODELS.keys()],
                    label="Model",
                    value=f"hf_{list(HUGGINGFACE_MODELS.keys())[0]}"
                )
                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Temperature")

            input_text = gr.Textbox(label="English text to translate", lines=3)
            output_txt = gr.Textbox(label="French translation")
            translate_btn = gr.Button("Translate")

            translate_btn.click(
                fn=translate,
                inputs=[direction_dropdown, model_dropdown, temp_slider, input_text],
                outputs=output_txt
            )

        # 📈 Onglets du visualizer
        with gr.Tab("📈 Model Performance by Temperature"):
            with gr.Row():
                model_dropdown_temp = gr.Dropdown(label="Select Model", choices=available_models, value=default_model)
                metric_dropdown_temp = gr.Dropdown(label="Select Metric", choices=available_metrics, value=default_metric)
            plot_output_temp = gr.Plot()
            model_dropdown_temp.change(visualizer.plot_model_performance_by_temperature, [model_dropdown_temp, metric_dropdown_temp], plot_output_temp)
            metric_dropdown_temp.change(visualizer.plot_model_performance_by_temperature, [model_dropdown_temp, metric_dropdown_temp], plot_output_temp)
            demo.load(visualizer.plot_model_performance_by_temperature, [model_dropdown_temp, metric_dropdown_temp], plot_output_temp)

        with gr.Tab("🏆 Overall Model Ranking"):
            metric_dropdown_ranking = gr.Dropdown(label="Select Ranking Metric", choices=["bleu", "rouge"], value="bleu")
            plot_output_ranking = gr.Plot()
            metric_dropdown_ranking.change(visualizer.plot_overall_model_ranking, [metric_dropdown_ranking], plot_output_ranking)
            demo.load(visualizer.plot_overall_model_ranking, [metric_dropdown_ranking], plot_output_ranking)

        with gr.Tab("🌡️ Temperature Impact Across Models"):
            metric_dropdown_temp_impact = gr.Dropdown(label="Select Metric", choices=available_metrics, value=default_metric)
            plot_output_temp_impact = gr.Plot()
            metric_dropdown_temp_impact.change(visualizer.plot_temperature_impact, [metric_dropdown_temp_impact], plot_output_temp_impact)
            demo.load(visualizer.plot_temperature_impact, [metric_dropdown_temp_impact], plot_output_temp_impact)

        with gr.Tab("🔍 Sample Translations"):
            with gr.Row():
                model_dropdown_samples = gr.Dropdown(label="Select Model", choices=available_models, value=default_model)
                temp_dropdown_samples = gr.Dropdown(label="Select Temperature", choices=available_temperatures, value=default_temp)
            num_samples_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Samples")
            sample_translations_output = gr.Markdown()

            def update_sample_translations(model, temp, num):
                return visualizer.get_sample_translations(model, temp, num)

            model_dropdown_samples.change(update_sample_translations, [model_dropdown_samples, temp_dropdown_samples, num_samples_slider], sample_translations_output)
            temp_dropdown_samples.change(update_sample_translations, [model_dropdown_samples, temp_dropdown_samples, num_samples_slider], sample_translations_output)
            num_samples_slider.change(update_sample_translations, [model_dropdown_samples, temp_dropdown_samples, num_samples_slider], sample_translations_output)
            demo.load(update_sample_translations, [model_dropdown_samples, temp_dropdown_samples, num_samples_slider], sample_translations_output)

        with gr.Tab("📖 Documentation"):
            gr.Markdown("""
            # 📚 Documentation — Tableau de bord de traduction automatique

            Bienvenue sur le tableau de bord de traduction et d'évaluation de modèles EN → FR.  
            Cette application vous permet à la fois **d’effectuer des traductions en temps réel** et **d’explorer les résultats d’évaluations quantitatives**.

            ## 🔄 Onglet « Traduction »
            - **Objectif :** Traduire un texte de l’anglais vers le français.
            - **Modèles disponibles :**
                - 🤗 HuggingFace :
                    - `google-t5/t5-base`
                    - `Helsinki-NLP/opus-mt-en-fr`
                    - `facebook/mbart-large-50-many-to-many-mmt`
                - 🦙 Ollama :
                    - `mistral`
                    - `llama3`
                    - `tinyllama:1.1b`
                    - `qwen2:0.5b`
            - **Température :** contrôle le niveau de créativité (0.0 = réponse déterministe, 1.0 = réponse plus variée).
            - **Sortie :** traduction automatique du texte saisi.

            ## 📈 Onglet « Model Performance by Temperature »
            - Visualisez l’évolution d’une métrique (BLEU, ROUGE, etc.) pour **un modèle spécifique** en fonction de la température.
            - Le graphique vous aide à voir comment la performance varie avec la créativité.

            ## 🏆 Onglet « Overall Model Ranking »
            - Affiche un **classement global des modèles** selon leur performance moyenne.
            - Métriques disponibles : BLEU et ROUGE.
            - Idéal pour identifier le meilleur modèle de façon globale.

            ## 🌡️ Onglet « Temperature Impact Across Models »
            - Compare **tous les modèles ensemble** à chaque niveau de température.
            - Permet d’identifier la température optimale en moyenne pour une métrique donnée.

            ## 🔍 Onglet « Sample Translations »
            - Affiche des exemples concrets de traductions générées :
                - Texte source
                - Traduction du modèle
                - Référence humaine
                - Scores BLEU & ROUGE
                - Temps de génération (en secondes)
            - Paramètres ajustables : modèle, température, nombre d'exemples.

            ## 📂 Fichiers requis
            Les résultats doivent être placés dans le dossier `results/` :
            - `results/comprehensive_results.json` (résultats bruts détaillés)
            - `results/evaluation_summary.json` (résumés de performance)

            ## 🚧 Fonctionnalités à venir
            - Traduction FR → EN
            - Téléchargement des résultats
            - Intégration de métriques supplémentaires (METEOR, TER, etc.)
            - Filtrage avancé des exemples

            ---
            **💡 Astuce :** Utilisez une température de `0.0` pour une traduction stable, et testez `0.7` ou `1.0` pour explorer la créativité des modèles type LLM.
            """)


if __name__ == "__main__":
    demo.launch(server_name=GRADIO_CONFIG["server_name"], server_port=GRADIO_CONFIG["server_port"], share=GRADIO_CONFIG["share"])
