import pytest
from unittest.mock import patch, MagicMock
from data_engineering_app.modules.config_validator import (
    validate_audit_config,
    validate_filter_config,
    validate_join_config,
    validate_all_configs
)

# Tests for validate_audit_config
def test_validate_audit_config():
    """Test the validate_audit_config function."""
    # Valid config
    valid_config = {
        "columns": {
            "id": {
                "type": "int",
                "required": True
            },
            "name": {
                "type": "string",
                "required": True
            }
        }
    }
    
    # Should not raise an exception
    validate_audit_config(valid_config)
    
    # Invalid config - missing columns
    invalid_config = {
        "error_mode": "warn"
    }
    
    with pytest.raises(ValueError):
        validate_audit_config(invalid_config)
    
    # Invalid config - empty columns
    invalid_config = {
        "columns": {}
    }
    
    with pytest.raises(ValueError):
        validate_audit_config(invalid_config)
    
    # Invalid config - missing column type
    invalid_config = {
        "columns": {
            "id": {
                "required": True
            }
        }
    }
    
    with pytest.raises(ValueError):
        validate_audit_config(invalid_config)

# Tests for validate_filter_config
def test_validate_filter_config():
    """Test the validate_filter_config function."""
    # Valid config with conditions
    valid_config = {
        "conditions": [
            "age > 30"
        ]
    }
    
    # Should not raise an exception
    validate_filter_config(valid_config)
    
    # Valid config with regex filters
    valid_config = {
        "regex_filters": {
            "name": "^[A-Z]"
        }
    }
    
    # Should not raise an exception
    validate_filter_config(valid_config)
    
    # Invalid config - no filter types
    invalid_config = {
        "other_param": "value"
    }
    
    with pytest.raises(ValueError):
        validate_filter_config(invalid_config)

# Tests for validate_join_config
def test_validate_join_config():
    """Test the validate_join_config function."""
    # Valid config
    valid_config = {
        "join_parameters": {
            "join_key": "id",
            "how": "inner"
        }
    }
    
    # Should not raise an exception
    validate_join_config(valid_config)
    
    # Invalid config - missing join_parameters
    invalid_config = {
        "filters": {
            "conditions": ["age > 30"]
        }
    }
    
    with pytest.raises(ValueError):
        validate_join_config(invalid_config)
    
    # Invalid config - missing join_key
    invalid_config = {
        "join_parameters": {
            "how": "inner"
        }
    }
    
    with pytest.raises(ValueError):
        validate_join_config(invalid_config)

# Tests for validate_all_configs
def test_validate_all_configs():
    """Test the validate_all_configs function."""
    # Valid configs
    audit_config = {
        "columns": {
            "id": {"type": "int", "required": True},
            "name": {"type": "string", "required": True}
        }
    }
    
    filter_config = {
        "conditions": ["age > 30"]
    }
    
    join_config = {
        "join_parameters": {
            "join_key": "id",
            "how": "inner"
        }
    }
    
    # Should not raise an exception
    validate_all_configs(audit_config, filter_config, join_config)
    
    # Test with invalid audit config
    invalid_audit = {}
    
    with pytest.raises(ValueError):
        validate_all_configs(invalid_audit, filter_config, join_config)
    
    # Test with invalid filter config
    invalid_filter = {}
    
    with pytest.raises(ValueError):
        validate_all_configs(audit_config, invalid_filter, join_config)
    
    # Test with invalid join config
    invalid_join = {}
    
    with pytest.raises(ValueError):
        validate_all_configs(audit_config, filter_config, invalid_join)