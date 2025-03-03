import os
import pytest
import pandas as pd
import numpy as np
import yaml
from unittest.mock import patch, mock_open

# Common fixtures that can be used across multiple test files

@pytest.fixture
def sample_data_path():
    """Return the path to the sample data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_engineering_app', 'data')

@pytest.fixture
def sample_config_path():
    """Return the path to the sample config directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_engineering_app', 'config')

@pytest.fixture
def mock_yaml_file():
    """Create a mock YAML file."""
    yaml_content = """
    test_config:
      param1: value1
      param2: 42
      nested:
        key1: value1
        key2: value2
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        with patch('os.path.exists', return_value=True):
            yield yaml_content

@pytest.fixture
def create_test_df():
    """Function to create test DataFrames with specified parameters."""
    def _create_df(rows=5, include_nulls=False, include_duplicates=False):
        data = {
            'id': list(range(1, rows + 1)),
            'name': [f'Person_{i}' for i in range(1, rows + 1)],
            'age': [20 + i * 5 for i in range(rows)],
            'score': [float(80 + i) for i in range(rows)],
            'date': pd.date_range(start='2020-01-01', periods=rows)
        }
        
        df = pd.DataFrame(data)
        
        # Add nulls if requested
        if include_nulls:
            df.loc[1, 'name'] = None
            df.loc[3, 'score'] = None
        
        # Add duplicates if requested
        if include_duplicates and rows > 2:
            df.loc[rows-1, 'id'] = df.loc[0, 'id']
            df.loc[rows-1, 'name'] = df.loc[0, 'name']
        
        return df
    
    return _create_df 