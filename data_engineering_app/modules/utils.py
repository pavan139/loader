# utils.py
import os
import logging
import yaml
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import functools

@functools.lru_cache(maxsize=32)
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with caching.
    """
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found: {config_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading YAML from {config_path}: {e}")
        raise

def validate_config_schema(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that a configuration dictionary contains required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required top-level keys
        
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

def get_file_type(file_path: str) -> str:
    """
    Determine file type from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type string ('csv', 'excel', 'parquet', etc.)
        
    Raises:
        ValueError: If the file type is not supported
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.csv':
        return 'csv'
    elif extension in ['.xlsx', '.xls']:
        return 'excel'
    elif extension == '.parquet':
        return 'parquet'
    elif extension == '.json':
        return 'json'
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def save_dataframe(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save a DataFrame to a file based on the file extension.
    
    Args:
        df: DataFrame to save
        file_path: Path where the file should be saved
        **kwargs: Additional arguments to pass to the save function
        
    Raises:
        ValueError: If the file type is not supported
    """
    try:
        file_type = get_file_type(file_path)
        
        if file_type == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_type == 'excel':
            df.to_excel(file_path, index=False, **kwargs)
        elif file_type == 'parquet':
            df.to_parquet(file_path, index=False, **kwargs)
        elif file_type == 'json':
            df.to_json(file_path, orient='records', **kwargs)
        else:
            raise ValueError(f"Unsupported output file type: {file_type}")
            
        logging.info(f"Saved DataFrame to {file_path}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {file_path}: {e}")
        raise

def log_dataframe_stats(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """
    Log basic statistics about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        label: Label to use in log messages
    """
    logging.info(f"{label} shape: {df.shape}")
    logging.info(f"{label} columns: {list(df.columns)}")
    
    # Log null counts
    null_counts = df.isnull().sum()
    if null_counts.any():
        logging.info(f"{label} null counts:")
        for col, count in null_counts[null_counts > 0].items():
            logging.info(f"  - '{col}': {count} nulls")
    else:
        logging.info(f"{label} has no null values")
    
    # Log data types
    logging.info(f"{label} data types:")
    for col, dtype in df.dtypes.items():
        logging.info(f"  - '{col}': {dtype}")

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Created directory: {directory_path}")

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate data quality metrics for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Basic counts
    metrics["row_count"] = len(df)
    metrics["column_count"] = len(df.columns)
    
    # Completeness (percentage of non-null values)
    metrics["completeness"] = float((df.count() / len(df)).mean())
    
    # Uniqueness (percentage of unique values per column)
    uniqueness = {}
    for col in df.columns:
        if len(df) > 0:
            uniqueness[col] = float(df[col].nunique() / len(df))
    metrics["uniqueness"] = uniqueness
    
    # Consistency (check for mixed data types)
    consistency = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, check percentage of values that can be converted to numbers
            try:
                consistency[col] = float(pd.to_numeric(df[col], errors='coerce').notna().mean())
            except:
                consistency[col] = 0.0
    metrics["consistency"] = consistency
    
    return metrics 
