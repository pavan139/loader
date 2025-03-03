# config_validator.py
import logging
from typing import Dict, List, Any, Optional
from .utils import validate_config_schema

def validate_audit_config(config: Dict[str, Any]) -> None:
    """
    Validate audit configuration structure.
    
    Args:
        config: Audit configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check for required top-level keys
    required_keys = ["columns"]
    validate_config_schema(config, required_keys)
    
    # Validate column configurations
    columns = config.get("columns", {})
    if not columns:
        raise ValueError("No columns defined in audit configuration")
    
    for col_name, col_config in columns.items():
        # Check column type
        col_type = col_config.get("type")
        if not col_type:
            raise ValueError(f"Column '{col_name}' is missing required 'type' attribute")
        
        # Validate type-specific configurations
        if col_type.lower() in ["date", "datetime"]:
            if "format" not in col_config and col_config.get("required", False):
                logging.warning(f"Date column '{col_name}' has no format specified")

def validate_filter_config(config: Dict[str, Any]) -> None:
    """
    Validate filter configuration structure.
    
    Args:
        config: Filter configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # At least one filter type should be present
    filter_types = ["conditions", "regex_filters", "null_filters", 
                   "duplicate_filters", "date_filters"]
    
    has_filters = any(config.get(filter_type) for filter_type in filter_types)
    if not has_filters:
        raise ValueError("Filter configuration contains no filter definitions")
    
    # Validate date filters if present
    date_filters = config.get("date_filters", {})
    for col, date_range in date_filters.items():
        if not isinstance(date_range, dict):
            raise ValueError(f"Date filter for '{col}' must be a dictionary")
        
        if "min" not in date_range and "max" not in date_range:
            raise ValueError(f"Date filter for '{col}' must specify at least 'min' or 'max'")

def validate_join_config(config: Dict[str, Any]) -> None:
    """
    Validate join configuration structure.
    
    Args:
        config: Join configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check for required sections
    required_keys = ["join_parameters"]
    validate_config_schema(config, required_keys)
    
    # Validate join parameters
    join_params = config.get("join_parameters", {})
    if "join_key" not in join_params:
        raise ValueError("Join configuration missing required 'join_key' parameter")
    
    # Validate join type if specified
    join_type = join_params.get("how", "inner")
    valid_join_types = ["inner", "left", "right", "outer", "cross"]
    if join_type not in valid_join_types:
        raise ValueError(f"Invalid join type '{join_type}'. Must be one of {valid_join_types}")
    
    # If filters are specified, validate them
    filters = config.get("filters")
    if filters:
        validate_filter_config(filters)

def validate_all_configs(audit_config: Dict[str, Any], 
                        filter_config: Dict[str, Any], 
                        join_config: Dict[str, Any]) -> None:
    """
    Validate all configurations at once.
    
    Args:
        audit_config: Audit configuration
        filter_config: Filter configuration
        join_config: Join configuration
        
    Raises:
        ValueError: If any configuration is invalid
    """
    validate_audit_config(audit_config)
    validate_filter_config(filter_config)
    validate_join_config(join_config)
    
    # Cross-validate configurations
    # For example, check that join keys exist in both audit configs
    join_key = join_config.get("join_parameters", {}).get("join_key")
    if join_key and join_key not in audit_config.get("columns", {}):
        raise ValueError(f"Join key '{join_key}' not defined in audit configuration") 