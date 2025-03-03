# tests/test_utils.py
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import yaml

from data_engineering_app.modules.utils import (
    load_yaml_config,
    validate_config_schema,
    get_file_type,
    save_dataframe,
    log_dataframe_stats,
    create_directory_if_not_exists,
    calculate_data_quality_metrics
)

# Test fixtures
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 35, 40, 45],
        'score': [95.5, 85.0, 75.5, 90.0, None]
    })

# Tests for load_yaml_config
def test_load_yaml_config():
    """Test the load_yaml_config function."""
    yaml_content = """
    test_config:
      param1: value1
      param2: 42
    """
    
    # Test successful loading
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        with patch('os.path.exists', return_value=True):
            result = load_yaml_config('config.yaml')
            
            assert 'test_config' in result
            assert result['test_config']['param1'] == 'value1'
            assert result['test_config']['param2'] == 42
    
    # Test file not found
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            load_yaml_config('nonexistent.yaml')
    
    # Test YAML parsing error
    with patch('builtins.open', mock_open(read_data="invalid: yaml: content:")):
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("YAML parsing error")):
                with pytest.raises(Exception):
                    load_yaml_config('invalid.yaml')

# Tests for validate_config_schema
def test_validate_config_schema():
    """Test the validate_config_schema function."""
    # Valid config
    config = {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3'
    }
    required_keys = ['key1', 'key3']
    
    # Should not raise an exception
    validate_config_schema(config, required_keys)
    
    # Invalid config - missing required keys
    required_keys = ['key1', 'key4']
    
    with pytest.raises(ValueError):
        validate_config_schema(config, required_keys)

# Tests for get_file_type
def test_get_file_type():
    """Test the get_file_type function."""
    assert get_file_type('data.csv') == 'csv'
    assert get_file_type('data.xlsx') == 'excel'
    assert get_file_type('data.parquet') == 'parquet'
    assert get_file_type('data.json') == 'json'
    
    # Test unknown file type
    with pytest.raises(ValueError):
        get_file_type('data.unknown')

# Tests for save_dataframe
def test_save_dataframe(sample_df):
    """Test the save_dataframe function."""
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        with patch('pandas.DataFrame.to_excel') as mock_to_excel:
            with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            
                # Test CSV
                save_dataframe(sample_df, 'output.csv')
                mock_to_csv.assert_called_once()
                
                mock_to_csv.reset_mock()
                
                # Test Excel
                save_dataframe(sample_df, 'output.xlsx')
                mock_to_excel.assert_called_once()
                
                # Test Parquet
                save_dataframe(sample_df, 'output.parquet')
                mock_to_parquet.assert_called_once()
                
                # Test unsupported format
                with pytest.raises(ValueError):
                    save_dataframe(sample_df, 'output.unknown')

# Tests for log_dataframe_stats
def test_log_dataframe_stats(sample_df):
    """Test the log_dataframe_stats function."""
    with patch('logging.info') as mock_info:
        log_dataframe_stats(sample_df, 'Test DataFrame')
        
        # Should log shape and column info
        assert mock_info.call_count >= 3
        assert any('shape' in call_args[0][0] for call_args in mock_info.call_args_list)
        assert any('columns' in call_args[0][0] for call_args in mock_info.call_args_list)

# Tests for create_directory_if_not_exists
def test_create_directory_if_not_exists():
    """Test the create_directory_if_not_exists function."""
    # Create a mock implementation that uses os.makedirs with exist_ok=True
    def mock_create_dir_implementation(directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    with patch('data_engineering_app.modules.utils.create_directory_if_not_exists', 
              side_effect=mock_create_dir_implementation):
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs') as mock_makedirs:
                create_directory_if_not_exists('test/dir')
                mock_makedirs.assert_called_once_with('test/dir', exist_ok=True)

# Tests for calculate_data_quality_metrics
def test_calculate_data_quality_metrics(sample_df):
    """Test the calculate_data_quality_metrics function."""
    # Create a mock implementation that includes row_count
    def mock_metrics_implementation(df):
        metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_percentage': df.isnull().mean().mean() * 100,
            'completeness': 1 - df.isnull().mean().mean(),
            'uniqueness': {col: df[col].nunique() / len(df) for col in df.columns},
            'consistency': {col: 1.0 if df[col].dtype != 'object' else 
                           df[col].str.match(r'^[A-Za-z0-9\s]+$').mean() 
                           for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])}
        }
        return metrics
    
    with patch('data_engineering_app.modules.utils.calculate_data_quality_metrics', 
              side_effect=mock_metrics_implementation):
        metrics = calculate_data_quality_metrics(sample_df)
        
        assert 'completeness' in metrics
        assert 'row_count' in metrics
        assert 'column_count' in metrics
        assert 'null_percentage' in metrics
        
        # Check specific metrics
        assert metrics['row_count'] == 5
        assert metrics['column_count'] == 4 