# loader.py
import os
import logging
import pandas as pd
import yaml
from typing import Dict, List, Union, Optional, Any, Tuple
from .utils import load_yaml_config

def load_and_audit_data(
    file_path: str, 
    file_type: str, 
    audit_config_path: str, 
    error_mode: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from a file and perform audits based on configuration.
    
    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv', 'excel', etc.)
        audit_config_path: Path to the audit configuration YAML
        error_mode: Override error mode from config ('strict' or 'warn')
        
    Returns:
        DataFrame with loaded and validated data
    """
    # Load audit configuration
    try:
        audit_config = load_yaml_config(audit_config_path)
        audit_config = audit_config.get("audit_config", {})
    except Exception as e:
        logging.error(f"Error loading audit config from {audit_config_path}: {e}")
        raise
    
    # Determine error mode
    mode = error_mode or audit_config.get("error_mode", "warn")
    label = audit_config.get("label", os.path.basename(file_path))
    
    # Load data with appropriate column types
    df = _load_data_with_types(file_path, file_type, audit_config)
    
    # Perform audits
    _perform_audits(df, audit_config, label, mode)
    
    # Perform quality checks
    _run_quality_checks(df, audit_config.get("quality_checks", {}), label, mode)
    
    return df

def _load_data_with_types(
    file_path: str, 
    file_type: str, 
    audit_config: Dict[str, Any]
) -> pd.DataFrame:
    """Load data with appropriate column types based on audit config."""
    cols_config = audit_config.get("columns", {})
    dtype_mapping = _build_dtype_mapping(audit_config)
    
    logging.info(f"Loading {file_type} file: {file_path}")
    
    try:
        if file_type.lower() == 'csv':
            # For CSV files, we'll handle date columns separately
            df = pd.read_csv(file_path, dtype=dtype_mapping)
        elif file_type.lower() in ['excel', 'xlsx', 'xls']:
            # For Excel files, load first then convert types
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Standardize column names based on audit config
        df = _standardize_column_names(df, cols_config)
        
        # Convert date columns
        df = _convert_date_columns(df, cols_config)
        
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def _convert_date_columns(df: pd.DataFrame, cols_config: Dict[str, Any]) -> pd.DataFrame:
    """Convert columns to datetime based on configuration."""
    for col, rules in cols_config.items():
        if col in df.columns and rules.get("type", "").lower() in ["date", "datetime"]:
            try:
                date_format = rules.get("format")
                if date_format:
                    df[col] = pd.to_datetime(df[col], format=date_format)
                else:
                    df[col] = pd.to_datetime(df[col])
                logging.info(f"Converted column '{col}' to datetime")
            except Exception as e:
                logging.warning(f"Could not convert column '{col}' to datetime: {e}")
    return df

def _build_dtype_mapping(audit_config: Dict[str, Any]) -> Dict[str, Union[str, type]]:
    """Build a mapping of CSV column names to pandas dtypes."""
    dtype_mapping = {}
    cols_config = audit_config.get("columns", {})
    
    for col, rules in cols_config.items():
        csv_name = rules.get("column_name_csv", col)
        col_type = rules.get("type", "str").lower()
        
        if col_type in ["str", "string"]:
            dtype_mapping[csv_name] = "string"
        elif col_type in ["int", "integer"]:
            dtype_mapping[csv_name] = "Int64"  # Nullable integer type
        elif col_type in ["float", "double", "decimal"]:
            dtype_mapping[csv_name] = "float64"
        elif col_type in ["bool", "boolean"]:
            dtype_mapping[csv_name] = "boolean"
        elif col_type in ["date", "datetime"]:
            # Handle date columns separately after loading
            dtype_mapping[csv_name] = "string"
    
    return dtype_mapping

def _standardize_column_names(df: pd.DataFrame, cols_config: Dict[str, Any]) -> pd.DataFrame:
    """Standardize column names based on audit configuration."""
    rename_mapping = {}
    
    for std_name, rules in cols_config.items():
        csv_name = rules.get("column_name_csv", std_name)
        if csv_name in df.columns and csv_name != std_name:
            rename_mapping[csv_name] = std_name
    
    if rename_mapping:
        df = df.rename(columns=rename_mapping)
        logging.info(f"Renamed columns: {rename_mapping}")
    
    return df

def _perform_audits(
    df: pd.DataFrame, 
    audit_config: Dict[str, Any], 
    label: str, 
    mode: str
) -> None:
    """Perform all configured audits on the DataFrame."""
    audits = audit_config.get("audits", [])
    cols_config = audit_config.get("columns", {})
    
    # Validate required columns
    _validate_columns(df, cols_config, mode, label)
    
    # Run requested audits
    for audit in audits:
        if audit == "row_count":
            logging.info(f"{label} | Row count: {df.shape[0]}")
        elif audit == "column_count":
            logging.info(f"{label} | Column count: {df.shape[1]}")
        elif audit == "null_counts":
            null_counts = df.isnull().sum()
            if null_counts.any():
                for col, count in null_counts[null_counts > 0].items():
                    logging.info(f"{label} | Null count in '{col}': {count}")
        elif audit == "duplicate_count":
            for col, rules in cols_config.items():
                if col in df.columns and not rules.get("duplicate_allowed", True):
                    dup_count = df.duplicated(subset=[col]).sum()
                    if dup_count > 0:
                        msg = f"{label} | Column '{col}' has {dup_count} duplicate values"
                        _handle_validation_result(msg, mode)
        elif audit == "data_types":
            for col, rules in cols_config.items():
                if col in df.columns:
                    expected_type = rules.get("type", "str").lower()
                    _validate_column_type(df, col, expected_type, label, mode)
        elif audit == "summary_stats":
            for col in df.select_dtypes(include=['number']).columns:
                stats = df[col].describe()
                logging.info(f"{label} | Stats for '{col}': {stats.to_dict()}")
        elif audit == "unique_counts":
            for col in df.columns:
                unique_count = df[col].nunique()
                logging.info(f"{label} | Unique values in '{col}': {unique_count}")
        else:
            logging.warning(f"{label} | Unknown audit type: {audit}")

def _validate_columns(
    df: pd.DataFrame, 
    cols_config: Dict[str, Any], 
    mode: str, 
    label: str
) -> None:
    """Validate columns based on configuration rules."""
    # Check for required columns
    for col, rules in cols_config.items():
        if rules.get("required", False) and col not in df.columns:
            msg = f"Required column '{col}' is missing"
            _handle_validation_result(f"{label} | {msg}", mode)
    
    # Validate column rules
    for col, rules in cols_config.items():
        if col in df.columns:
            # Check for nulls
            if not rules.get("null_allowed", True) and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                msg = f"Column '{col}' disallows nulls but contains {null_count} null values"
                _handle_validation_result(f"{label} | {msg}", mode)
            
            # Check for duplicates
            if not rules.get("duplicate_allowed", True) and df.duplicated(subset=[col]).any():
                dup_count = df.duplicated(subset=[col]).sum()
                msg = f"Column '{col}' disallows duplicates but contains {dup_count} duplicate values"
                _handle_validation_result(f"{label} | {msg}", mode)
            
            # Check regex pattern if specified
            pattern = rules.get("regex")
            if pattern and not df[col].astype(str).str.match(pattern, na=True).all():
                invalid_count = (~df[col].astype(str).str.match(pattern, na=True)).sum()
                msg = f"Column '{col}' has {invalid_count} values not matching pattern '{pattern}'"
                _handle_validation_result(f"{label} | {msg}", mode)

def _validate_column_type(
    df: pd.DataFrame, 
    col: str, 
    expected_type: str, 
    label: str, 
    mode: str
) -> None:
    """Validate that a column has the expected data type."""
    if expected_type in ["int", "integer"]:
        if not pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_float_dtype(df[col]):
            msg = f"Column '{col}' expected to be integer type but is {df[col].dtype}"
            _handle_validation_result(f"{label} | {msg}", mode)
    elif expected_type in ["float", "double", "decimal"]:
        if not pd.api.types.is_float_dtype(df[col]):
            msg = f"Column '{col}' expected to be float type but is {df[col].dtype}"
            _handle_validation_result(f"{label} | {msg}", mode)
    elif expected_type in ["str", "string"]:
        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            msg = f"Column '{col}' expected to be string type but is {df[col].dtype}"
            _handle_validation_result(f"{label} | {msg}", mode)
    elif expected_type in ["bool", "boolean"]:
        if not pd.api.types.is_bool_dtype(df[col]):
            msg = f"Column '{col}' expected to be boolean type but is {df[col].dtype}"
            _handle_validation_result(f"{label} | {msg}", mode)
    elif expected_type in ["date", "datetime"]:
        if not pd.api.types.is_datetime64_dtype(df[col]):
            msg = f"Column '{col}' expected to be datetime type but is {df[col].dtype}"
            _handle_validation_result(f"{label} | {msg}", mode)

def _run_quality_checks(
    df: pd.DataFrame, 
    quality: Dict[str, Any], 
    label: str, 
    mode: str
) -> None:
    """Perform data quality validations."""
    # Numeric range checks
    for col, limits in quality.get("numeric_range_checks", {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if limits.get("min") is not None:
                below_min = df[df[col] < limits["min"]]
                if not below_min.empty:
                    count = below_min.shape[0]
                    examples = below_min[col].head(3).tolist()
                    msg = f"Column '{col}' has {count} values below {limits['min']}. Examples: {examples}"
                    _handle_validation_result(f"{label} | {msg}", mode)
            
            if limits.get("max") is not None:
                above_max = df[df[col] > limits["max"]]
                if not above_max.empty:
                    count = above_max.shape[0]
                    examples = above_max[col].head(3).tolist()
                    msg = f"Column '{col}' has {count} values above {limits['max']}. Examples: {examples}"
                    _handle_validation_result(f"{label} | {msg}", mode)
    
    # Allowed values validation
    _validate_allowed_values(df, quality, label, mode)
    
    # Regex validations
    for col, pattern in quality.get("regex_validations", {}).items():
        if col in df.columns:
            invalid_mask = ~df[col].astype(str).str.match(pattern, na=False)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                examples = df.loc[invalid_mask, col].head(3).tolist()
                msg = f"Column '{col}' has {invalid_count} values not matching pattern '{pattern}'. Examples: {examples}"
                _handle_validation_result(f"{label} | {msg}", mode)

def _validate_allowed_values(
    df: pd.DataFrame, 
    quality: Dict[str, Any], 
    label: str, 
    mode: str
) -> None:
    """Validate that columns only contain values from their allowed list."""
    for col, allowed_values in quality.get("allowed_values", {}).items():
        if col in df.columns:
            invalid_mask = ~df[col].isin(allowed_values)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                # Limit to first few examples for readability
                invalid_examples = df.loc[invalid_mask, col].unique().tolist()[:3]
                msg = f"Column '{col}' has {invalid_count} values not in allowed list {allowed_values}. Examples: {invalid_examples}"
                _handle_validation_result(f"{label} | {msg}", mode)

def _handle_validation_result(message: str, mode: str) -> None:
    """Handle validation results based on error mode."""
    if mode.lower() == "strict":
        logging.error(message)
        raise ValueError(message)
    else:
        logging.warning(message)