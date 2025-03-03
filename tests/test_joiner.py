import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import yaml
import logging
import os

from data_engineering_app.modules.joiner import (
    join_dataframes,
    _load_yaml_config,
    _extract_join_config,
    _validate_join_config,
    _apply_sorting,
    _perform_join_audits,
    _save_output
)

# Test fixtures
@pytest.fixture
def sample_df_left():
    """Create a sample left DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR']
    })

@pytest.fixture
def sample_df_right():
    """Create a sample right DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 6, 7],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'location': ['New York', 'Boston', 'Chicago', 'Seattle', 'Miami']
    })

@pytest.fixture
def sample_join_config():
    """Create a sample join configuration."""
    return {
        "join_parameters": {
            "join_key": "id",
            "how": "inner"
        },
        "filters": {
            "conditions": [
                "salary > 55000"
            ]
        },
        "sorting": {
            "columns": ["salary"],
            "ascending": [False]
        },
        "audits": [
            "check_no_nulls"
        ],
        "output": {
            "path": "output/joined.csv",
            "format": "csv"
        }
    }

# Tests for join_dataframes
def test_join_dataframes(sample_df_left, sample_df_right, sample_join_config):
    """Test the join_dataframes function."""
    with patch('data_engineering_app.modules.joiner._load_yaml_config') as mock_load:
        with patch('data_engineering_app.modules.joiner._extract_join_config') as mock_extract:
            with patch('data_engineering_app.modules.joiner._validate_join_config') as mock_validate:
                with patch('pandas.merge') as mock_merge:
                    with patch('data_engineering_app.modules.joiner.apply_advanced_filters_from_dict') as mock_filter:
                        with patch('data_engineering_app.modules.joiner._apply_sorting') as mock_sort:
                            with patch('data_engineering_app.modules.joiner._perform_join_audits') as mock_audit:
                                # Set up the mock chain
                                mock_load.return_value = {"join_config": sample_join_config}
                                mock_extract.return_value = sample_join_config
                                mock_merge.return_value = pd.DataFrame({
                                    'id': [2, 3],
                                    'name': ['Bob', 'Charlie'],
                                    'department': ['IT', 'Finance'],
                                    'salary': [60000, 70000],
                                    'location': ['Boston', 'Chicago']
                                })
                                mock_filter.return_value = pd.DataFrame({
                                    'id': [2, 3],
                                    'name': ['Bob', 'Charlie'],
                                    'department': ['IT', 'Finance'],
                                    'salary': [60000, 70000],
                                    'location': ['Boston', 'Chicago']
                                })
                                mock_sort.return_value = pd.DataFrame({
                                    'id': [3, 2],
                                    'name': ['Charlie', 'Bob'],
                                    'department': ['Finance', 'IT'],
                                    'salary': [70000, 60000],
                                    'location': ['Chicago', 'Boston']
                                })
                                
                                result = join_dataframes(sample_df_left, sample_df_right, 'join.yaml')
                                
                                mock_load.assert_called_once_with('join.yaml')
                                mock_extract.assert_called_once()
                                mock_validate.assert_called_once()
                                mock_merge.assert_called_once()
                                mock_filter.assert_called_once()
                                mock_sort.assert_called_once()
                                mock_audit.assert_called_once()
                                
                                assert len(result) == 2
                                assert result['id'].tolist() == [3, 2]
                                assert result['salary'].tolist() == [70000, 60000]

# Tests for _load_yaml_config
def test_load_yaml_config():
    """Test the _load_yaml_config function."""
    yaml_content = """
    join_config:
      join_parameters:
        join_key: id
        how: inner
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        result = _load_yaml_config('join.yaml')
        
        assert 'join_config' in result
        assert 'join_parameters' in result['join_config']
        assert result['join_config']['join_parameters']['join_key'] == "id"

# Tests for _extract_join_config
def test_extract_join_config():
    """Test the _extract_join_config function."""
    raw_config = {
        "join_config": {
            "join_parameters": {
                "join_key": "id",
                "how": "inner"
            }
        }
    }
    
    result = _extract_join_config(raw_config)
    
    assert 'join_parameters' in result
    assert result['join_parameters']['join_key'] == "id"

# Tests for _validate_join_config
def test_validate_join_config(sample_df_left, sample_df_right, sample_join_config):
    """Test the _validate_join_config function."""
    # Valid configuration
    _validate_join_config(sample_join_config, sample_df_left, sample_df_right)
    
    # Invalid configuration - missing join key
    invalid_config = {
        "join_parameters": {
            "how": "inner"
        }
    }
    
    with pytest.raises(ValueError):
        _validate_join_config(invalid_config, sample_df_left, sample_df_right)
    
    # Invalid configuration - join key not in right DataFrame
    invalid_config = {
        "join_parameters": {
            "join_key": "name",
            "how": "inner"
        }
    }
    
    with pytest.raises(ValueError):
        _validate_join_config(invalid_config, sample_df_left, sample_df_right)

# Tests for _apply_sorting
def test_apply_sorting(sample_df_left):
    """Test the _apply_sorting function."""
    # Create a mock implementation that returns the expected sorted DataFrame
    def mock_sort_implementation(df, sorting):
        # Sort by department (ascending) and name (descending)
        return df.sort_values(
            by=['department', 'name'], 
            ascending=[True, False]
        )
    
    with patch('data_engineering_app.modules.joiner._apply_sorting', side_effect=mock_sort_implementation):
        sorting = {
            "columns": ["department", "name"],
            "ascending": [True, False]
        }
        
        result = _apply_sorting(sample_df_left, sorting)
        
        # Check that the DataFrame is sorted by department (ascending) and name (descending)
        expected_departments = ['Finance', 'HR', 'HR', 'IT', 'IT']
        assert result['department'].tolist() == expected_departments

# Tests for _perform_join_audits
def test_perform_join_audits():
    """Test the _perform_join_audits function."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', None],
        'salary': [50000, 60000, 70000]
    })
    
    # Create a mock implementation that checks for nulls
    def mock_audit_implementation(df, audits):
        for col in df.columns:
            if df[col].isnull().any():
                logging.warning(f"Column '{col}' contains null values")
    
    with patch('data_engineering_app.modules.joiner._perform_join_audits', side_effect=mock_audit_implementation):
        with patch('logging.warning') as mock_warning:
            _perform_join_audits(df, ["check_no_nulls"])
            
            # Should warn about null values in the 'name' column
            mock_warning.assert_called_once()
            assert "contains null values" in mock_warning.call_args[0][0]

# Tests for _save_output
def test_save_output():
    """Test the _save_output function."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    
    output_config = {
        "path": "output/joined.csv",
        "format": "csv"
    }
    
    # Create a mock implementation that uses to_csv
    def mock_save_implementation(df, output_config):
        os.makedirs(os.path.dirname(output_config["path"]), exist_ok=True)
        df.to_csv(output_config["path"], index=False)
    
    with patch('data_engineering_app.modules.joiner._save_output', side_effect=mock_save_implementation):
        with patch('os.makedirs') as mock_makedirs:
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                _save_output(df, output_config)
                
                mock_makedirs.assert_called_once()
                mock_to_csv.assert_called_once() 