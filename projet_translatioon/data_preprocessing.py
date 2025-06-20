"""
Data preprocessing module for the translation demo
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import re
from sklearn.model_selection import train_test_split

from config import DATASET_CONFIG, DATA_DIR
from utils import preprocess_text, validate_translation_pair, timer_decorator

logger = logging.getLogger(__name__)

class TranslationDataProcessor:
    """
    Class for processing translation datasets
    """
    
    def __init__(self, config: dict = None):
        self.config = config or DATASET_CONFIG
        self.source_lang = self.config["source_lang"]
        self.target_lang = self.config["target_lang"]
        self.source_column = self.config["source_column"]
        self.target_column = self.config["target_column"]
        
    @timer_decorator
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the translation dataset
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with translation pairs
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded: {len(df)} samples")
            
            # Verify required columns exist
            required_cols = [self.source_column, self.target_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text entries
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that might interfere with translation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', text)
        
        # Remove very short or very long texts
        if len(text) < 3 or len(text) > 500:
            return ""
        
        return text
    
    @timer_decorator
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting dataset preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text columns
        processed_df[self.source_column] = processed_df[self.source_column].apply(self.clean_text)
        processed_df[self.target_column] = processed_df[self.target_column].apply(self.clean_text)
        
        # Remove rows with empty texts
        initial_length = len(processed_df)
        processed_df = processed_df[
            (processed_df[self.source_column] != "") & 
            (processed_df[self.target_column] != "")
        ]
        
        logger.info(f"Removed {initial_length - len(processed_df)} empty entries")
        
        # Remove duplicates
        initial_length = len(processed_df)
        processed_df = processed_df.drop_duplicates(subset=[self.source_column, self.target_column])
        logger.info(f"Removed {initial_length - len(processed_df)} duplicate entries")
        
        # Validate translation pairs
        valid_mask = processed_df.apply(
            lambda row: validate_translation_pair(
                row[self.source_column], 
                row[self.target_column]
            ), 
            axis=1
        )
        
        processed_df = processed_df[valid_mask]
        logger.info(f"Final dataset size after preprocessing: {len(processed_df)}")
        
        return processed_df
    
    def analyze_dataset(self, df: pd.DataFrame) -> dict:
        """
        Analyze dataset characteristics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(df)
        analysis['source_lang'] = self.source_lang
        analysis['target_lang'] = self.target_lang
        
        # Text length statistics
        source_lengths = df[self.source_column].str.len()
        target_lengths = df[self.target_column].str.len()
        
        analysis['source_length_stats'] = {
            'mean': source_lengths.mean(),
            'median': source_lengths.median(),
            'std': source_lengths.std(),
            'min': source_lengths.min(),
            'max': source_lengths.max()
        }
        
        analysis['target_length_stats'] = {
            'mean': target_lengths.mean(),
            'median': target_lengths.median(),
            'std': target_lengths.std(),
            'min': target_lengths.min(),
            'max': target_lengths.max()
        }
        
        # Word count statistics
        source_word_counts = df[self.source_column].str.split().str.len()
        target_word_counts = df[self.target_column].str.split().str.len()
        
        analysis['source_word_stats'] = {
            'mean': source_word_counts.mean(),
            'median': source_word_counts.median(),
            'std': source_word_counts.std()
        }
        
        analysis['target_word_stats'] = {
            'mean': target_word_counts.mean(),
            'median': target_word_counts.median(),
            'std': target_word_counts.std()
        }
        
        logger.info("Dataset analysis completed")
        return analysis
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        test_size = self.config.get("test_size", 0.1)
        val_size = self.config.get("validation_size", 0.1)
        random_state = self.config.get("random_state", 42)
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size_adjusted, 
            random_state=random_state
        )
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples") 
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def sample_for_demo(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Sample dataset for demo purposes
        
        Args:
            df: Input DataFrame
            sample_size: Number of samples (defaults to config value)
            
        Returns:
            Sampled DataFrame
        """
        if sample_size is None:
            sample_size = self.config.get("sample_size", 1000)
        
        if len(df) <= sample_size:
            logger.info(f"Dataset size ({len(df)}) is smaller than sample size ({sample_size}), using full dataset")
            return df
        
        # Stratified sampling to maintain diversity
        sampled_df = df.sample(
            n=sample_size, 
            random_state=self.config.get("random_state", 42)
        )
        
        logger.info(f"Dataset sampled: {len(sampled_df)} samples")
        return sampled_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, output_dir: Path):
        """
        Save processed datasets to files
        
        Args:
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            output_dir: Output directory
        """
        output_dir.mkdir(exist_ok=True)
        
        # Save datasets
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        logger.info(f"Processed datasets saved to {output_dir}")
    
    def load_processed_data(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed datasets
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        try:
            train_df = pd.read_csv(data_dir / "train.csv")
            val_df = pd.read_csv(data_dir / "val.csv")
            test_df = pd.read_csv(data_dir / "test.csv")
            
            logger.info("Processed datasets loaded successfully")
            return train_df, val_df, test_df
            
        except FileNotFoundError as e:
            logger.error(f"Processed data files not found: {e}")
            raise

def main():
    """
    Main function for data preprocessing
    """
    # Initialize processor
    processor = TranslationDataProcessor()
    
    # Load raw data
    data_file = DATA_DIR / "sample_fr_en_clean.csv"  # Adjust filename as needed
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please download the dataset and place it in the data/ directory")
        return
    
    # Load and preprocess
    raw_df = processor.load_data(data_file)
    processed_df = processor.preprocess_dataset(raw_df)
    
    # Analyze dataset
    analysis = processor.analyze_dataset(processed_df)
    logger.info(f"Dataset analysis: {analysis}")
    
    # Sample for demo
    demo_df = processor.sample_for_demo(processed_df)
    
    # Split dataset
    train_df, val_df, test_df = processor.split_dataset(demo_df)
    
    # Save processed data
    processor.save_processed_data(train_df, val_df, test_df, DATA_DIR)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()