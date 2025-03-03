# transformer.py
import logging
import pandas as pd
import yaml
from typing import Dict, List, Union, Optional, Any

def apply_advanced_filters_from_config(
    df: pd.DataFrame, 
    filter_config_path: str
) -> pd.DataFrame:
    """
    Apply advanced filters to a DataFrame based on a configuration file.
    
    Args:
        df: Input DataFrame
        filter_config_path: Path to the filter configuration YAML
        
    Returns:
        Filtered DataFrame
    """
    try:
        raw_config = _load_yaml_config(filter_config_path)
        config = raw_config.get("filter_config", {})
    except Exception as e:
        logging.error(f"Error loading filter config from {filter_config_path}: {e}")
        raise
    
    return apply_advanced_filters_from_dict(df, config)

def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading YAML from {config_path}: {e}")
        raise

def apply_advanced_filters_from_dict(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply advanced filters to a DataFrame based on a configuration dictionary.
    
    Args:
        df: Input DataFrame
        config: Filter configuration dictionary
        
    Returns:
        Filtered DataFrame
    """
    original_row_count = df.shape[0]
    logging.info(f"Starting filtering with {original_row_count} rows")
    
    # Apply query conditions: filter rows based on pandas query expressions.
    df = _apply_query_conditions(df, config.get("conditions", []))
    
    # Apply regex filters: filter rows based on regex patterns.
    df = _apply_regex_filters(df, config.get("regex_filters", {}))
    
    # Apply null filters: drop rows with nulls in specified columns.
    df = _apply_null_filters(df, config.get("null_filters", []))
    
    # Apply duplicate filters: remove duplicates based on specified columns.
    df = _apply_duplicate_filters(df, config.get("duplicate_filters", []))
    
    # Apply date filters: filter rows based on a date range.
    df = _apply_date_filters(df, config.get("date_filters", {}))
    
    final_row_count = df.shape[0]
    rows_removed = original_row_count - final_row_count
    logging.info(f"Filtering complete: {rows_removed} rows removed ({final_row_count} remaining)")
    
    return df

def _apply_query_conditions(df: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
    """Apply query conditions to filter the DataFrame."""
    for condition in conditions:
        try:
            before = df.shape[0]
            df = df.query(condition)
            after = df.shape[0]
            logging.info(f"Applied condition '{condition}'; rows before: {before}, after: {after}")
        except Exception as e:
            logging.error(f"Error applying condition '{condition}': {e}")
            raise
    return df

def _apply_regex_filters(df: pd.DataFrame, regex_filters: Dict[str, str]) -> pd.DataFrame:
    """Apply regex filters to the DataFrame."""
    # Create a single mask for all regex filters to avoid multiple DataFrame copies
    if not regex_filters:
        return df
        
    mask = pd.Series(True, index=df.index)
    for col, pattern in regex_filters.items():
        if col in df.columns:
            try:
                col_mask = df[col].astype(str).str.match(pattern, na=False)
                before = mask.sum()
                mask &= col_mask
                after = mask.sum()
                logging.info(f"Applied regex filter on '{col}' with pattern '{pattern}'; rows before: {before}, after: {after}")
            except Exception as e:
                logging.error(f"Error applying regex filter on '{col}': {e}")
                raise
        else:
            logging.warning(f"Column '{col}' not found for regex filtering.")
    
    return df[mask]

def _apply_null_filters(df: pd.DataFrame, null_filters: List[str]) -> pd.DataFrame:
    """Drop rows with nulls in specified columns."""
    if not null_filters:
        return df
        
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in null_filters if col in df.columns]
    
    if valid_cols:
        before = df.shape[0]
        df = df.dropna(subset=valid_cols)
        after = df.shape[0]
        logging.info(f"Applied null filters on columns {valid_cols}; rows before: {before}, after: {after}")
    
    # Log warnings for columns not found
    for col in null_filters:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found for null filtering.")
    
    return df

def _apply_duplicate_filters(df: pd.DataFrame, duplicate_filters: List[str]) -> pd.DataFrame:
    """Remove duplicates based on specified columns."""
    if not duplicate_filters:
        return df
        
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in duplicate_filters if col in df.columns]
    
    if valid_cols:
        before = df.shape[0]
        df = df.drop_duplicates(subset=valid_cols)
        after = df.shape[0]
        logging.info(f"Applied duplicate filters on columns {valid_cols}; rows before: {before}, after: {after}")
    
    # Log warnings for columns not found
    for col in duplicate_filters:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found for duplicate filtering.")
    
    return df

def _apply_date_filters(df: pd.DataFrame, date_filters: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Filter rows based on date ranges."""
    for col, drange in date_filters.items():
        if col in df.columns:
            try:
                # Convert column to datetime if it's not already
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Apply min/start date filter if specified
                min_date = None
                if "min" in drange:
                    min_date = pd.to_datetime(drange["min"])
                elif "start" in drange:
                    min_date = pd.to_datetime(drange["start"])
                
                if min_date is not None:
                    before = df.shape[0]
                    df = df[df[col] >= min_date]
                    after = df.shape[0]
                    logging.info(f"Applied min date filter on '{col}' (>= {min_date}); rows before: {before}, after: {after}")
                
                # Apply max/end date filter if specified
                max_date = None
                if "max" in drange:
                    max_date = pd.to_datetime(drange["max"])
                elif "end" in drange:
                    max_date = pd.to_datetime(drange["end"])
                
                if max_date is not None:
                    before = df.shape[0]
                    df = df[df[col] <= max_date]
                    after = df.shape[0]
                    logging.info(f"Applied max date filter on '{col}' (<= {max_date}); rows before: {before}, after: {after}")
                
            except Exception as e:
                logging.error(f"Error applying date filter on '{col}': {e}")
                raise
        else:
            logging.warning(f"Column '{col}' not found for date filtering.")
    
    return df