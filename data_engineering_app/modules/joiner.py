# joiner.py
import logging
import os  # Added import for os
import pandas as pd
import yaml
from typing import Dict, List, Union, Optional, Any
from .transformer import apply_advanced_filters_from_dict

def join_dataframes(
    df_left: pd.DataFrame, 
    df_right: pd.DataFrame, 
    join_config_path: str
) -> pd.DataFrame:
    """
    Join two DataFrames based on a join configuration YAML file.
    
    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        join_config_path: Path to the join configuration YAML
        
    Returns:
        Joined DataFrame
    """
    try:
        raw_config = _load_yaml_config(join_config_path)
        config = _extract_join_config(raw_config)
    except Exception as e:
        logging.error(f"Error loading join config from {join_config_path}: {e}")
        raise

    # Validate join configuration
    _validate_join_config(config, df_left, df_right)

    # Extract join parameters
    join_params = config.get("join_parameters", {})
    join_key = join_params.get("join_key")
    how = join_params.get("how", "inner")

    # Perform the join
    joined = pd.merge(df_left, df_right, on=join_key, how=how, suffixes=("_left", "_right"))
    logging.info(f"Joined DataFrames on '{join_key}' using '{how}' join; shape: {joined.shape}")

    # Apply advanced filters if provided
    filters_config = config.get("filters")
    if filters_config:
        joined = apply_advanced_filters_from_dict(joined, filters_config)
        logging.info(f"Applied advanced join filters; shape: {joined.shape}")

    # Apply sorting if provided
    sorting = config.get("sorting")
    if sorting:
        joined = _apply_sorting(joined, sorting)

    # Perform audits if requested
    _perform_join_audits(joined, config.get("audits", []))

    # Save output if requested
    output_config = config.get("output", {})
    if output_config.get("save_to_file", False):
        _save_output(joined, output_config)

    return joined

def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading YAML from {config_path}: {e}")
        raise

def _extract_join_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract join configuration from raw config."""
    join_config = raw_config.get("join_config", {})
    if not join_config:
        raise ValueError("Join configuration not found in config file")
    return join_config

def _validate_join_config(
    config: Dict[str, Any], 
    df_left: pd.DataFrame, 
    df_right: pd.DataFrame
) -> None:
    """Validate join configuration against DataFrames."""
    join_params = config.get("join_parameters", {})
    join_key = join_params.get("join_key")
    
    if not join_key:
        raise ValueError("Join key not specified in configuration")
    
    if join_key not in df_left.columns:
        raise ValueError(f"Join key '{join_key}' not found in left DataFrame")
    
    if join_key not in df_right.columns:
        raise ValueError(f"Join key '{join_key}' not found in right DataFrame")
    
    how = join_params.get("how", "inner")
    valid_join_types = ["inner", "left", "right", "outer", "cross"]
    if how not in valid_join_types:
        raise ValueError(f"Invalid join type '{how}'. Must be one of {valid_join_types}")

def _apply_sorting(df: pd.DataFrame, sorting: Dict[str, Any]) -> pd.DataFrame:
    """Apply sorting to the DataFrame."""
    # Support both single column (backward compatibility) and multiple columns
    sort_cols = sorting.get("columns")
    if sort_cols is None:
        sort_col = sorting.get("column")
        if sort_col in df.columns:
            sort_cols = [sort_col]
            ascending = sorting.get("ascending", True)
            df = df.sort_values(by=sort_col, ascending=ascending)
            logging.info(f"Sorted joined DataFrame by '{sort_col}' (ascending={ascending})")
        else:
            logging.warning(f"Sort column '{sort_col}' not found in joined DataFrame.")
            return df
    else:
        # Check if all columns exist in the DataFrame
        missing_cols = [col for col in sort_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Sort columns {missing_cols} not found in joined DataFrame.")
            # Filter out missing columns
            sort_cols = [col for col in sort_cols if col in df.columns]
            if not sort_cols:
                return df
        
        # Get ascending parameter (can be boolean or list of booleans)
        ascending = sorting.get("ascending", True)
        
        # Sort the DataFrame
        df = df.sort_values(by=sort_cols, ascending=ascending)
        logging.info(f"Sorted joined DataFrame by {sort_cols} (ascending={ascending})")
    
    return df

def _perform_join_audits(df: pd.DataFrame, audits: List[str]) -> None:
    """Perform audits on the joined DataFrame."""
    for audit in audits:
        if audit == "row_count":
            logging.info(f"Joined DataFrame row count: {df.shape[0]}")
        elif audit == "column_count":
            logging.info(f"Joined DataFrame column count: {df.shape[1]}")
        elif audit == "null_counts":
            null_counts = df.isnull().sum()
            if null_counts.any():
                for col, count in null_counts[null_counts > 0].items():
                    logging.info(f"Null count in '{col}': {count}")
        elif audit == "check_no_nulls":
            null_counts = df.isnull().sum()
            if null_counts.any():
                for col, count in null_counts[null_counts > 0].items():
                    logging.warning(f"Column '{col}' contains null values")
        elif audit == "duplicate_count":
            for col in df.columns:
                dup_count = df.duplicated(subset=[col]).sum()
                if dup_count > 0:
                    logging.info(f"Column '{col}' has {dup_count} duplicate values")
        else:
            logging.warning(f"Unknown audit type: {audit}")

def _save_output(df: pd.DataFrame, output_config: Dict[str, Any]) -> None:
    """Save the joined DataFrame to a file."""
    # Support both file_path and path keys for backward compatibility
    file_path = output_config.get("file_path") or output_config.get("path")
    if not file_path:
        logging.warning("Output file path not specified, skipping save")
        return
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Determine file type from extension
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df.to_csv(file_path, index=False)
        elif file_ext in ['xlsx', 'xls']:
            df.to_excel(file_path, index=False)
        elif file_ext == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            logging.warning(f"Unsupported output file type: {file_ext}")
            return
            
        logging.info(f"Saved joined DataFrame to {file_path}")
    except Exception as e:
        logging.error(f"Error saving output to {file_path}: {e}")
        raise