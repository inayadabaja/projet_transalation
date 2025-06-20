import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """
    Class to load and visualize translation evaluation results using Gradio and Plotly.
    """
    def __init__(self):
        self.all_results = {}
        self.summary_results = {}
        logger.info("ResultsVisualizer initialized.")

    def load_results(self, results_dir: Path):
        """
        Loads the comprehensive evaluation results and summary from the specified directory.

        Args:
            results_dir: Path to the directory containing the JSON results.
        """
        comprehensive_path = results_dir / "comprehensive_results.json"
        summary_path = results_dir / "evaluation_summary.json"

        if not comprehensive_path.exists() or not summary_path.exists():
            logger.error(f"Results files not found in {results_dir}. Please run evaluation first.")
            self.all_results = {}
            self.summary_results = {}
            return

        try:
            with open(comprehensive_path, 'r', encoding='utf-8') as f:
                self.all_results = json.load(f)
            logger.info(f"Loaded comprehensive results from {comprehensive_path}")

            with open(summary_path, 'r', encoding='utf-8') as f:
                self.summary_results = json.load(f)
            logger.info(f"Loaded evaluation summary from {summary_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from results files: {e}")
            self.all_results = {}
            self.summary_results = {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading results: {e}")
            self.all_results = {}
            self.summary_results = {}

    def get_available_models(self) -> List[str]:
        """Returns a list of all model names available in the results."""
        return sorted(list(self.all_results.keys())) if self.all_results else []

    def get_available_metrics(self) -> List[str]:
        """
        Returns a list of all unique metric keys found in the results.
        Prioritizes common aggregate metrics.
        """
        if not self.all_results:
            return []

        # Find metrics from the first available sample translation
        for model_name in self.all_results:
            model_data = self.all_results[model_name]
            for temp_str in model_data.get("temperature_results", {}):
                temp_results = model_data["temperature_results"][temp_str]
                if temp_results and "sample_translations" in temp_results and temp_results["sample_translations"]:
                    first_sample = temp_results["sample_translations"][0]
                    if "scores" in first_sample:
                        all_metrics = list(first_sample["scores"].keys())
                        # Prioritize overall BLEU and ROUGE
                        prioritized_metrics = []
                        if "bleu" in all_metrics:
                            prioritized_metrics.append("bleu")
                        if "rouge" in all_metrics:
                            prioritized_metrics.append("rouge")
                        remaining_metrics = sorted([m for m in all_metrics if m not in ["bleu", "rouge"]])
                        return prioritized_metrics + remaining_metrics
        return []

    def get_available_temperatures(self) -> List[float]:
        """Returns a list of all unique temperatures tested across models."""
        temperatures = set()
        for model_name, model_data in self.all_results.items():
            for temp_str in model_data.get("temperature_results", {}):
                try:
                    temperatures.add(float(temp_str))
                except ValueError:
                    logger.warning(f"Could not convert temperature '{temp_str}' to float for model {model_name}.")
        return sorted(list(temperatures))

    def plot_model_performance_by_temperature(self, model_name: str, metric: str) -> go.Figure:
        """
        Generates a line plot showing a specific model's performance (metric)
        across different temperatures.
        """
        if not self.all_results or model_name not in self.all_results:
            return go.Figure().add_annotation(text="No data loaded or model not found.", showarrow=False)

        model_data = self.all_results[model_name]
        temperatures = []
        scores = []
        score_stds = []

        for temp_str, temp_summary in sorted(model_data.get("temperature_results", {}).items(), key=lambda item: float(item[0])):
            try:
                temp = float(temp_str)
                if metric in temp_summary:
                    temperatures.append(temp)
                    scores.append(temp_summary[metric])
                    # Try to get standard deviation if available, otherwise default to 0
                    if f"{metric}_std" in temp_summary:
                        score_stds.append(temp_summary[f"{metric}_std"])
                    else:
                        score_stds.append(0) # Default to 0 if std not recorded for this metric
            except ValueError:
                continue # Skip if temperature string is not a valid float

        if not temperatures:
            return go.Figure().add_annotation(text=f"No data for metric '{metric}' or temperatures for {model_name}.", showarrow=False)

        df = pd.DataFrame({
            "Temperature": temperatures,
            metric.replace("_", " ").title(): scores,
            "Std Dev": score_stds
        })

        fig = px.line(df, x="Temperature", y=metric.replace("_", " ").title(),
                      title=f'{model_name} {metric.replace("_", " ").title()} vs. Temperature',
                      markers=True)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(hovermode="x unified") # For better hover experience

        # Add error bars if standard deviation is available
        if any(s > 0 for s in score_stds):
             fig.add_trace(go.Scatter(
                 name="Upper Bound",
                 x=df["Temperature"],
                 y=df[metric.replace("_", " ").title()] + df["Std Dev"],
                 mode='lines',
                 line=dict(width=0),
                 showlegend=False
             ))
             fig.add_trace(go.Scatter(
                 name="Lower Bound",
                 x=df["Temperature"],
                 y=df[metric.replace("_", " ").title()] - df["Std Dev"],
                 mode='lines',
                 line=dict(width=0),
                 fillcolor='rgba(0,100,80,0.2)',
                 fill='tonexty',
                 showlegend=False
             ))


        return fig

    def plot_overall_model_ranking(self, metric_type: str) -> go.Figure:
        """
        Generates a bar chart showing models ranked by a chosen metric type (e.g., 'bleu', 'rouge').
        """
        if not self.summary_results or "model_rankings" not in self.summary_results:
            return go.Figure().add_annotation(text="Evaluation summary not loaded.", showarrow=False)

        ranking_key = f"by_{metric_type}"
        if ranking_key not in self.summary_results["model_rankings"]:
            return go.Figure().add_annotation(text=f"Ranking for '{metric_type}' not found.", showarrow=False)

        ranking_data = self.summary_results["model_rankings"][ranking_key]
        models = [item["model"] for item in ranking_data]
        scores = [item["score"] for item in ranking_data]

        df = pd.DataFrame({"Model": models, f"Average {metric_type.title()} Score": scores})

        fig = px.bar(df, x="Model", y=f"Average {metric_type.title()} Score",
                     title=f"Overall Model Ranking by Average {metric_type.title()} Score",
                     color=f"Average {metric_type.title()} Score",
                     color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def plot_temperature_impact(self, metric: str) -> go.Figure:
        """
        Génère un graphique en ligne montrant la performance moyenne
        de tous les modèles à différentes températures pour une métrique donnée.
        """

        if not self.summary_results or "temperature_analysis" not in self.summary_results:
            return go.Figure().add_annotation(text="Evaluation summary not loaded.", showarrow=False)

        temp_analysis = self.summary_results["temperature_analysis"]
        temperatures = []
        avg_scores = []
        score_stds = []

        for temp_str in sorted(temp_analysis.keys(), key=lambda x: float(x)):
            data = temp_analysis[temp_str]
            if f"avg_{metric}" in data:
                temperatures.append(float(temp_str))
                avg_scores.append(data[f"avg_{metric}"])
                score_stds.append(data.get(f"{metric}_std", 0))

        if not temperatures:
            return go.Figure().add_annotation(text=f"No aggregate temperature data for metric '{metric}'.", showarrow=False)

        # Correspondance sûre entre métrique et nom de colonne
        metric_column_map = {
            "bleu": "Average Bleu Score",
            "rouge": "Average Rouge Score"
            # ajoute ici d'autres métriques si besoin
        }

        y_col = metric_column_map.get(metric.lower())
        if y_col is None:
            return go.Figure().add_annotation(text=f"Unsupported metric '{metric}'.", showarrow=False)

        df = pd.DataFrame({
            "Temperature": temperatures,
            y_col: avg_scores,
            "Std Dev": score_stds
        })

        fig = px.line(df, x="Temperature", y=y_col,
                    title=f"{y_col} Across All Models by Temperature",
                    markers=True)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(hovermode="x unified")

        # Ajouter les barres d’erreur (borne supérieure et inférieure)
        if any(s > 0 for s in score_stds):
            df["Upper Bound"] = df[y_col] + df["Std Dev"]
            df["Lower Bound"] = df[y_col] - df["Std Dev"]

            fig.add_trace(go.Scatter(
                name="Upper Bound",
                x=df["Temperature"],
                y=df["Upper Bound"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                name="Lower Bound",
                x=df["Temperature"],
                y=df["Lower Bound"],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                showlegend=False
            ))

        return fig



    def get_sample_translations(self, model_name: str, temperature: float, num_samples: int = 5) -> str:
        """
        Retrieves and formats a few sample translations for a given model and temperature.
        """
        if not self.all_results or model_name not in self.all_results:
            return "No data loaded or model not found."

        model_data = self.all_results[model_name]
        temp_str = str(temperature)

        if temp_str not in model_data.get("temperature_results", {}):
            return f"No data for temperature {temperature} for model {model_name}."

        temp_results = model_data["temperature_results"][temp_str]
        sample_translations = temp_results.get("sample_translations", [])

        if not sample_translations:
            return f"No sample translations found for {model_name} at temperature {temperature}."

        output = f"### Sample Translations for {model_name} (Temperature: {temperature})\n\n"
        for i, sample in enumerate(sample_translations[:num_samples]):
            output += f"#### Sample {i+1}\n"
            output += f"**Source:** {sample.get('source', 'N/A')}\n"
            output += f"**Reference:** {sample.get('reference', 'N/A')}\n"
            output += f"**Translation:** {sample.get('translation', 'N/A')}\n"
            output += f"**BLEU:** {sample.get('scores', {}).get('bleu', 0.0):.4f}\n"
            output += f"**ROUGE:** {sample.get('scores', {}).get('rouge', 0.0):.4f}\n"
            output += f"**Time (s):** {sample.get('translation_time', 0.0):.4f}\n\n"
        return output

# --- Gradio Interface Setup ---

# Initialize the visualizer and load results
results_dir_path = Path("results") # Assuming 'results' directory exists relative to this script
visualizer = ResultsVisualizer()
visualizer.load_results(results_dir_path)

# Get available options for dropdowns
available_models = visualizer.get_available_models()
available_metrics = visualizer.get_available_metrics()
available_temperatures = visualizer.get_available_temperatures()

# Ensure there are default selections if data is available
default_model = available_models[0] if available_models else None
default_metric = available_metrics[0] if available_metrics else None
default_temp = available_temperatures[0] if available_temperatures else None

with gr.Blocks(title="Translation Model Evaluation Dashboard") as demo:
    gr.Markdown("# Translation Model Evaluation Dashboard")
    gr.Markdown("Explore the performance of different translation models across various metrics and temperatures.")

    with gr.Tab("Model Performance by Temperature"):
        with gr.Row():
            model_dropdown_temp = gr.Dropdown(
                label="Select Model",
                choices=available_models,
                value=default_model,
                interactive=True
            )
            metric_dropdown_temp = gr.Dropdown(
                label="Select Metric",
                choices=available_metrics,
                value=default_metric,
                interactive=True
            )
        plot_output_temp = gr.Plot(label="Model Performance vs. Temperature")
        model_dropdown_temp.change(
            fn=visualizer.plot_model_performance_by_temperature,
            inputs=[model_dropdown_temp, metric_dropdown_temp],
            outputs=plot_output_temp
        )
        metric_dropdown_temp.change(
            fn=visualizer.plot_model_performance_by_temperature,
            inputs=[model_dropdown_temp, metric_dropdown_temp],
            outputs=plot_output_temp
        )
        # Initial plot load
        demo.load(
            fn=visualizer.plot_model_performance_by_temperature,
            inputs=[model_dropdown_temp, metric_dropdown_temp],
            outputs=plot_output_temp
        )

    with gr.Tab("Overall Model Ranking"):
        metric_dropdown_ranking = gr.Dropdown(
            label="Select Ranking Metric",
            choices=["bleu", "rouge"], # Only these for overall ranking summary
            value="bleu",
            interactive=True
        )
        plot_output_ranking = gr.Plot(label="Overall Model Ranking")
        metric_dropdown_ranking.change(
            fn=visualizer.plot_overall_model_ranking,
            inputs=[metric_dropdown_ranking],
            outputs=plot_output_ranking
        )
        # Initial plot load
        demo.load(
            fn=visualizer.plot_overall_model_ranking,
            inputs=[metric_dropdown_ranking],
            outputs=plot_output_ranking
        )

    with gr.Tab("Temperature Impact Across Models"):
        metric_dropdown_temp_impact = gr.Dropdown(
            label="Select Metric",
            choices=available_metrics,
            value=default_metric,
            interactive=True
        )
        plot_output_temp_impact = gr.Plot(label="Average Performance by Temperature (All Models)")
        metric_dropdown_temp_impact.change(
            fn=visualizer.plot_temperature_impact,
            inputs=[metric_dropdown_temp_impact],
            outputs=plot_output_temp_impact
        )
        # Initial plot load
        demo.load(
            fn=visualizer.plot_temperature_impact,
            inputs=[metric_dropdown_temp_impact],
            outputs=plot_output_temp_impact
        )

    with gr.Tab("Sample Translations"):
        with gr.Row():
            model_dropdown_samples = gr.Dropdown(
                label="Select Model",
                choices=available_models,
                value=default_model,
                interactive=True
            )
            temp_dropdown_samples = gr.Dropdown(
                label="Select Temperature",
                choices=available_temperatures,
                value=default_temp,
                interactive=True
            )
        num_samples_slider = gr.Slider(
            minimum=1, maximum=10, value=5, step=1, label="Number of Samples"
        )
        sample_translations_output = gr.Markdown()

        def update_sample_translations(model, temp, num):
            return visualizer.get_sample_translations(model, temp, num)

        model_dropdown_samples.change(
            fn=update_sample_translations,
            inputs=[model_dropdown_samples, temp_dropdown_samples, num_samples_slider],
            outputs=sample_translations_output
        )
        temp_dropdown_samples.change(
            fn=update_sample_translations,
            inputs=[model_dropdown_samples, temp_dropdown_samples, num_samples_slider],
            outputs=sample_translations_output
        )
        num_samples_slider.change(
            fn=update_sample_translations,
            inputs=[model_dropdown_samples, temp_dropdown_samples, num_samples_slider],
            outputs=sample_translations_output
        )
        # Initial load
        demo.load(
            fn=update_sample_translations,
            inputs=[model_dropdown_samples, temp_dropdown_samples, num_samples_slider],
            outputs=sample_translations_output
        )


if __name__ == "__main__":
    if not available_models:
        logger.warning("No evaluation results found. Please run 'main.py' first to generate results.")
    demo.launch()