"""
Comprehensive evaluation module for translation models
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import sacrebleu

# Import evaluation metrics
from evaluate import load
from sacrebleu import BLEU
from rouge_score import rouge_scorer

# Import model interfaces
import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from config import (
    HUGGINGFACE_MODELS, OLLAMA_MODELS, TEMPERATURE_SETTINGS, 
    EVALUATION_CONFIG, TRANSLATION_PROMPTS, PERFORMANCE_CONFIG
)
from utils import timer_decorator, save_results, format_translation_output

logger = logging.getLogger(__name__)

class TranslationEvaluator:
    """
    Comprehensive evaluation class for translation models
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize evaluation metrics
        self.bleu_metric = load("bleu")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Model caches
        self.hf_models = {}
        self.hf_tokenizers = {}
        
    def load_huggingface_model(self, model_config: Dict) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Load HuggingFace model and tokenizer
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_config["model_name"]
        
        if model_name not in self.hf_models:
            logger.info(f"Loading HuggingFace model: {model_name}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.to(self.device)
                
                self.hf_tokenizers[model_name] = tokenizer
                self.hf_models[model_name] = model
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
                raise
        
        return self.hf_models[model_name], self.hf_tokenizers[model_name]
    
    def translate_with_huggingface(self, text: str, model_config: Dict, 
                                 temperature: float = 0.0) -> str:
        """
        Translate text using HuggingFace model
        
        Args:
            text: Source text to translate
            model_config: Model configuration
            temperature: Sampling temperature
            
        Returns:
            Translated text
        """
        try:
            model, tokenizer = self.load_huggingface_model(model_config)
            
            # Prepare input based on model type
            if "t5" in model_config["model_name"].lower():
                input_text = f"translate English to French: {text}"
            else:
                input_text = text
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=model_config["max_length"], 
                             truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                if temperature == 0.0:
                    # Greedy decoding
                    outputs = model.generate(
                        **inputs,
                        max_length=model_config["max_length"],
                        num_beams=4,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                else:
                    # Sampling with temperature
                    outputs = model.generate(
                        **inputs,
                        max_length=model_config["max_length"],
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
            
            # Decode output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error in HuggingFace translation: {e}")
            return f"Translation error: {str(e)}"
    
    def translate_with_ollama(self, text: str, model_config: Dict,
                            temperature: float = 0.0, source_lang: str = "en",
                            target_lang: str = "fr") -> str:
        """
        Translate text using Ollama model

        Args:
            text: Source text to translate
            model_config: Model configuration (must include 'model_name')
            temperature: Sampling temperature (0.0 = deterministic)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        try:
            model_name = model_config.get("model_name")
            if not model_name:
                raise ValueError("Missing 'model_name' in model_config.")

            # Nettoyage du texte
            text = text.strip()

            # Choix du prompt
            if source_lang == "en" and target_lang == "fr":
                prompt_template = TRANSLATION_PROMPTS.get("en_to_fr", "Translate English to French:\n{text}")
            elif source_lang == "fr" and target_lang == "en":
                prompt_template = TRANSLATION_PROMPTS.get("fr_to_en", "Translate French to English:\n{text}")
            else:
                prompt_template = f"Translate from {source_lang} to {target_lang}:\n{{text}}"

            prompt = prompt_template.format(text=text)
            logger.debug(f"Prompt: {prompt}")

            # Appel à l'API Ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": TRANSLATION_PROMPTS.get("system_prompt", "")},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "top_p": 0.9 if temperature > 0 else 1.0,
                    "num_predict": 256
                }
            )

            # 📌 Nouvelle extraction du contenu
            if hasattr(response, "message") and hasattr(response.message, "content"):
                translation = response.message.content.strip()
                return translation
            else:
                logger.error(f"Unexpected Ollama response structure: {response}")
                return "Translation error: Unexpected response structure from Ollama."

        except Exception as e:
            logger.error(f"Error in Ollama translation: {e}")
            return f"Translation error: {str(e)}"
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate BLEU scores from BLEU-1 to BLEU-4 using sacrebleu.sentence_bleu
        
        Args:
            reference: Reference translation string
            hypothesis: Model translation string
            
        Returns:
            Dictionary with BLEU scores (floats between 0 and 1)
        """
        try:
            bleu_scores = {}
            for n in range(1, 5):
                bleu = BLEU(max_ngram_order=n, smooth_method="exp", effective_order=True)
                score = bleu.sentence_score(hypothesis, [reference])
                bleu_scores[f"bleu_{n}"] = score.score / 100.0
            bleu_scores["bleu"] = bleu_scores["bleu_4"]
            return bleu_scores
        
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}    
    def calculate_rouge_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        
        Args:
            reference: Reference translation
            hypothesis: Model translation
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            
            rouge_scores = {}
            for metric, score in scores.items():
                rouge_scores[f"{metric}_f"] = score.fmeasure
                rouge_scores[f"{metric}_p"] = score.precision
                rouge_scores[f"{metric}_r"] = score.recall
            
            # Overall ROUGE score (average of F1 scores)
            rouge_scores["rouge"] = np.mean([
                rouge_scores["rouge1_f"],
                rouge_scores["rouge2_f"], 
                rouge_scores["rougeL_f"]
            ])
            
            return rouge_scores
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            return {"rouge": 0.0}
    
    def evaluate_single_translation(self, source: str, reference: str, 
                                  model_name: str, model_config: Dict,
                                  temperature: float, model_type: str) -> Dict:
        """
        Evaluate a single translation
        
        Args:
            source: Source text
            reference: Reference translation
            model_name: Name of the model
            model_config: Model configuration
            temperature: Temperature setting
            model_type: Type of model ('hf' or 'ollama')
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        start_time = time.time()
        try:
            if model_type == "hf":
                translation = self.translate_with_huggingface(source, model_config, temperature)
            elif model_type == "ollama":
                # Passer source et target lang si disponibles
                source_lang = model_config.get("source_lang", "en")
                target_lang = model_config.get("target_lang", "fr")
                translation = self.translate_with_ollama(source, model_config, temperature, source_lang, target_lang)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            bleu_scores = self.calculate_bleu_score(reference, translation)
            rouge_scores = self.calculate_rouge_score(reference, translation)
            all_scores = {**bleu_scores, **rouge_scores}
            translation_time = time.time() - start_time
            
            return {
                "source": source,
                "reference": reference,
                "translation": translation,
                "scores": all_scores,
                "translation_time": translation_time,
                "model": model_name,
                "temperature": temperature
            }
        except Exception as e:
            logger.error(f"Error evaluating translation: {e}")
            return {
                "source": source,
                "reference": reference,
                "translation": f"Error: {str(e)}",
                "scores": {"bleu": 0.0, "rouge": 0.0},
                "translation_time": time.time() - start_time,
                "model": model_name,
                "temperature": temperature
            }

    
    @timer_decorator
    def evaluate_model_batch(self, test_data: pd.DataFrame, model_name: str, 
                           model_config: Dict, model_type: str, 
                           temperatures: List[float] = None) -> Dict:
        """
        Evaluate model across different temperatures on batch of data
        
        Args:
            test_data: Test dataset
            model_name: Name of the model
            model_config: Model configuration
            model_type: Type of model ('hf' or 'ollama')
            temperatures: List of temperatures to test
            
        Returns:
            Dictionary with evaluation results
        """
        if temperatures is None:
            temperatures = TEMPERATURE_SETTINGS
        
        logger.info(f"Evaluating {model_name} on {len(test_data)} samples")
        
        results = {
            "model_name": model_name,
            "model_type": model_type,
            "temperature_results": {},
            "overall_stats": {}
        }
        
        for temperature in temperatures:
            logger.info(f"Testing {model_name} with temperature {temperature}")
            
            temp_results = []
            
            # Process each sample
            for idx, row in tqdm(test_data.iterrows(), total=len(test_data), 
                               desc=f"{model_name} T={temperature}"):
                
                source_text = row['en']  # Adjust column name as needed
                reference_text = row['fr']  # Adjust column name as needed
                
                result = self.evaluate_single_translation(
                    source_text, reference_text, model_name, 
                    model_config, temperature, model_type
                )
                
                temp_results.append(result)
            
            # Aggregate results for this temperature
            bleu_scores = [r["scores"]["bleu"] for r in temp_results]
            rouge_scores = [r["scores"]["rouge"] for r in temp_results]
            translation_times = [r["translation_time"] for r in temp_results]
            
            temp_summary = {
                "bleu": np.mean(bleu_scores),
                "bleu_std": np.std(bleu_scores),
                "rouge": np.mean(rouge_scores),
                "rouge_std": np.std(rouge_scores),
                "avg_time": np.mean(translation_times),
                "total_time": np.sum(translation_times),
                "sample_translations": temp_results[:5]  # Store first 5 for inspection
            }
            
            results["temperature_results"][str(temperature)] = temp_summary
            
            logger.info(f"T={temperature}: BLEU={temp_summary['bleu']:.4f}, ROUGE={temp_summary['rouge']:.4f}")
        
        # Calculate overall statistics
        all_bleu = []
        all_rouge = []
        for temp_data in results["temperature_results"].values():
            all_bleu.append(temp_data["bleu"])
            all_rouge.append(temp_data["rouge"])
        
        results["overall_stats"] = {
            "best_bleu_temp": temperatures[np.argmax(all_bleu)],
            "best_bleu_score": max(all_bleu),
            "best_rouge_temp": temperatures[np.argmax(all_rouge)],
            "best_rouge_score": max(all_rouge),
            "avg_bleu": np.mean(all_bleu),
            "avg_rouge": np.mean(all_rouge)
        }
        
        return results
    
    def run_comprehensive_evaluation(self, test_data: pd.DataFrame, 
                               output_dir: Path,
                               save_partial: bool = False) -> Dict:
        """
        Run comprehensive evaluation on all models
        
        Args:
            test_data: Test dataset
            output_dir: Directory to save results
            save_partial: If True, save results after each model evaluation
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        
        all_results = {}
        
        # Evaluate HuggingFace models
        logger.info("Evaluating HuggingFace models...")
        for model_key, model_config in HUGGINGFACE_MODELS.items():
            try:
                model_name = f"hf_{model_key}"
                results = self.evaluate_model_batch(
                    test_data, model_name, model_config, "hf"
                )
                all_results[model_name] = results
                
                # Save individual model results if requested
                if save_partial:
                    save_results(results, f"{model_name}_results.json", output_dir)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_key}: {e}")
                continue
        
        # Evaluate Ollama models
        logger.info("Evaluating Ollama models...")
        for model_key, model_config in OLLAMA_MODELS.items():
            try:
                model_name = f"ollama_{model_key}"
                results = self.evaluate_model_batch(
                    test_data, model_name, model_config, "ollama"
                )
                all_results[model_name] = results
                
                # Save individual model results if requested
                if save_partial:
                    save_results(results, f"{model_name}_results.json", output_dir)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_key}: {e}")
                continue
        
        # Save comprehensive results
        save_results(all_results, "comprehensive_results.json", output_dir)
        
        # Generate summary report
        summary = self.generate_summary_report(all_results)
        save_results(summary, "evaluation_summary.json", output_dir)
        
        logger.info("Comprehensive evaluation completed!")
        return all_results


    
    def generate_summary_report(self, all_results: Dict) -> Dict:
        """
        Generate summary report from evaluation results
        
        Args:
            all_results: Dictionary with all evaluation results
            
        Returns:
            Summary report dictionary
        """
        summary = {
            "evaluation_overview": {
                "total_models": len(all_results),
                "hf_models": len([k for k in all_results.keys() if k.startswith("hf_")]),
                "ollama_models": len([k for k in all_results.keys() if k.startswith("ollama_")]),
                "temperatures_tested": TEMPERATURE_SETTINGS
            },
            "best_performers": {},
            "model_rankings": {},
            "temperature_analysis": {}
        }
        
        # Find best performers
        best_bleu_score = 0
        best_bleu_model = ""
        best_rouge_score = 0
        best_rouge_model = ""
        
        model_scores = {}
        
        for model_name, results in all_results.items():
            if "overall_stats" in results:
                stats = results["overall_stats"]
                
                # Track best performers
                if stats["best_bleu_score"] > best_bleu_score:
                    best_bleu_score = stats["best_bleu_score"]
                    best_bleu_model = model_name
                
                if stats["best_rouge_score"] > best_rouge_score:
                    best_rouge_score = stats["best_rouge_score"]
                    best_rouge_model = model_name
                
                # Store model scores for ranking
                model_scores[model_name] = {
                    "avg_bleu": stats["avg_bleu"],
                    "avg_rouge": stats["avg_rouge"],
                    "best_bleu": stats["best_bleu_score"],
                    "best_rouge": stats["best_rouge_score"]
                }
        
        summary["best_performers"] = {
            "bleu": {"model": best_bleu_model, "score": best_bleu_score},
            "rouge": {"model": best_rouge_model, "score": best_rouge_score}
        }
        
        # Rank models by average performance
        bleu_ranking = sorted(model_scores.items(), 
                            key=lambda x: x[1]["avg_bleu"], reverse=True)
        rouge_ranking = sorted(model_scores.items(), 
                             key=lambda x: x[1]["avg_rouge"], reverse=True)
        
        summary["model_rankings"] = {
            "by_bleu": [{"model": m, "score": s["avg_bleu"]} for m, s in bleu_ranking],
            "by_rouge": [{"model": m, "score": s["avg_rouge"]} for m, s in rouge_ranking]
        }
        
        # Temperature analysis
        temp_analysis = {}
        for temp in TEMPERATURE_SETTINGS:
            temp_str = str(temp)
            temp_bleu_scores = []
            temp_rouge_scores = []
            
            for results in all_results.values():
                if temp_str in results.get("temperature_results", {}):
                    temp_data = results["temperature_results"][temp_str]
                    temp_bleu_scores.append(temp_data["bleu"])
                    temp_rouge_scores.append(temp_data["rouge"])
            
            if temp_bleu_scores:
                temp_analysis[temp_str] = {
                    "avg_bleu": np.mean(temp_bleu_scores),
                    "avg_rouge": np.mean(temp_rouge_scores),
                    "bleu_std": np.std(temp_bleu_scores),
                    "rouge_std": np.std(temp_rouge_scores)
                }
        
        summary["temperature_analysis"] = temp_analysis
        
        return summary

def main():
    """
    Main function for running evaluation on all models and saving results
    """
    from data_preprocessing import TranslationDataProcessor
    
    # Initialize components
    processor = TranslationDataProcessor()
    evaluator = TranslationEvaluator()
    
    # Load test data
    try:
        _, _, test_df = processor.load_processed_data(Path("data"))
        logger.info(f"Loaded test data: {len(test_df)} samples")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run data preprocessing first.")
        return
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Use a subset for quick testing (optionnel : enlever .head pour tout tester)
    test_subset = test_df.head(50)  # Ajuste à None ou supprime pour tout le dataset
    
    # Run evaluation on all models and save results incrementally
    all_results = evaluator.run_comprehensive_evaluation(test_subset, results_dir, save_partial=True)
    
    logger.info("Evaluation completed! Check the 'results' directory for detailed results.")

if __name__ == "__main__":
    main()