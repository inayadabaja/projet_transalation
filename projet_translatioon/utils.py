"""
Utility functions for the translation demo project
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import LOGGING_CONFIG, DATASET_CONFIG, EVALUATION_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["filename"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the translation dataset from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with translation pairs
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully: {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for translation
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Basic cleaning
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

def split_dataset(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1, 
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state
    )
    
    logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def sample_dataset(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """
    Sample a subset of the dataset for demo purposes
    
    Args:
        df: Input DataFrame
        sample_size: Number of samples to select
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    logger.info(f"Dataset sampled: {len(sampled_df)} samples")
    
    return sampled_df

def save_results(results: Dict, filename: str, results_dir: Path):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary containing results
        filename: Output filename
        results_dir: Directory to save results
    """
    try:
        results_dir.mkdir(exist_ok=True)
        file_path = results_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def load_results(filename: str, results_dir: Path) -> Optional[Dict]:
    """
    Load evaluation results from JSON file
    
    Args:
        filename: Input filename
        results_dir: Directory containing results
        
    Returns:
        Dictionary with results or None if file doesn't exist
    """
    try:
        file_path = results_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Results file not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def create_comparison_plot(results: Dict, metric: str = "bleu") -> go.Figure:
    """
    Create interactive comparison plot for model performance
    
    Args:
        results: Dictionary containing evaluation results
        metric: Metric to plot ('bleu' or 'rouge')
        
    Returns:
        Plotly figure
    """
    data = []
    
    for model_name, model_results in results.items():
        for temp, scores in model_results.get('temperature_results', {}).items():
            if metric in scores:
                data.append({
                    'Model': model_name,
                    'Temperature': float(temp),
                    'Score': scores[metric],
                    'Type': 'HuggingFace' if 'hf_' in model_name else 'Ollama'
                })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        logger.warning(f"No data found for metric: {metric}")
        return go.Figure()
    
    fig = px.line(
        df, x='Temperature', y='Score', color='Model',
        line_dash='Type',
        title=f'{metric.upper()} Scores by Temperature and Model',
        labels={'Score': f'{metric.upper()} Score', 'Temperature': 'Temperature'}
    )
    
    fig.update_layout(
        xaxis=dict(tickvals=[0.0, 0.2, 0.5, 0.8, 1.0]),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_heatmap(results: Dict, metric: str = "bleu") -> go.Figure:
    """
    Create heatmap showing model performance across temperatures
    
    Args:
        results: Dictionary containing evaluation results
        metric: Metric to visualize
        
    Returns:
        Plotly figure
    """
    models = []
    temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
    scores_matrix = []
    
    for model_name, model_results in results.items():
        models.append(model_name)
        scores_row = []
        
        for temp in temperatures:
            temp_str = str(temp)
            score = model_results.get('temperature_results', {}).get(temp_str, {}).get(metric, 0)
            scores_row.append(score)
        
        scores_matrix.append(scores_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix,
        x=[str(t) for t in temperatures],
        y=models,
        colorscale='RdYlBu_r',
        colorbar=dict(title=f"{metric.upper()} Score")
    ))
    
    fig.update_layout(
        title=f'{metric.upper()} Scores Heatmap',
        xaxis_title='Temperature',
        yaxis_title='Model'
    )
    
    return fig

def format_translation_output(source_text: str, translation: str, 
                            model_name: str, temperature: float, 
                            scores: Optional[Dict] = None) -> str:
    """
    Format translation output for display
    
    Args:
        source_text: Original text
        translation: Translated text
        model_name: Name of the model used
        temperature: Temperature setting
        scores: Optional evaluation scores
        
    Returns:
        Formatted output string
    """
    output = f"""
### 🤖 Model: {model_name}
**Temperature:** {temperature}

**Original:** {source_text}
**Translation:** {translation}
"""
    
    if scores:
        output += "\n**📊 Scores:**\n"
        for metric, score in scores.items():
            output += f"- {metric.upper()}: {score:.4f}\n"
    
    return output

def calculate_average_scores(results: Dict) -> Dict:
    """
    Calculate average scores across all models and temperatures
    
    Args:
        results: Dictionary containing evaluation results
        
    Returns:
        Dictionary with average scores
    """
    all_scores = {'bleu': [], 'rouge': []}
    
    for model_results in results.values():
        for temp_results in model_results.get('temperature_results', {}).values():
            for metric in all_scores.keys():
                if metric in temp_results:
                    all_scores[metric].append(temp_results[metric])
    
    avg_scores = {}
    for metric, scores in all_scores.items():
        if scores:
            avg_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
    
    return avg_scores

def timer_decorator(func):
    """
    Decorator to measure function execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper

def validate_translation_pair(source: str, target: str) -> bool:
    """
    Validate if a translation pair is suitable for evaluation
    
    Args:
        source: Source text
        target: Target text
        
    Returns:
        True if valid, False otherwise
    """
    if not source or not target:
        return False
    
    if len(source.strip()) < 3 or len(target.strip()) < 3:
        return False
    
    # Check if texts are too similar (might be duplicates)
    if source.lower() == target.lower():
        return False
    
    return True