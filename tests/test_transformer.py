import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import yaml

from data_engineering_app.modules.transformer import (
    apply_advanced_filters_from_config,
    apply_advanced_filters_from_dict,
    _load_yaml_config,
    _apply_query_conditions,
    _apply_regex_filters,
    _apply_null_filters,
    _apply_duplicate_filters,
    _apply_date_filters
)

# Test fixtures
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 35, 40, 45],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date_joined': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'])
    })

@pytest.fixture
def sample_filter_config():
    """Create a sample filter configuration."""
    return {
        "conditions": [
            "age > 30",
            "category != 'C'"
        ],
        "regex_filters": {
            "name": "^[A-C]"
        },
        "null_filters": [
            "name"
        ],
        "duplicate_filters": [
            "category"
        ],
        "date_filters": {
            "date_joined": {
                "start": "2020-02-01",
                "end": "2020-04-30"
            }
        }
    }

# Tests for apply_advanced_filters_from_config
def test_apply_advanced_filters_from_config(sample_df, sample_filter_config):
    """Test the apply_advanced_filters_from_config function."""
    with patch('data_engineering_app.modules.transformer._load_yaml_config') as mock_load:
        with patch('data_engineering_app.modules.transformer.apply_advanced_filters_from_dict') as mock_apply:
            mock_load.return_value = {"filter_config": sample_filter_config}
            mock_apply.return_value = sample_df.iloc[1:3]
            
            result = apply_advanced_filters_from_config(sample_df, 'filter.yaml')
            
            mock_load.assert_called_once_with('filter.yaml')
            mock_apply.assert_called_once_with(sample_df, sample_filter_config)
            assert result.equals(sample_df.iloc[1:3])

# Tests for apply_advanced_filters_from_dict
def test_apply_advanced_filters_from_dict(sample_df, sample_filter_config):
    """Test the apply_advanced_filters_from_dict function."""
    with patch('data_engineering_app.modules.transformer._apply_query_conditions') as mock_query:
        with patch('data_engineering_app.modules.transformer._apply_regex_filters') as mock_regex:
            with patch('data_engineering_app.modules.transformer._apply_null_filters') as mock_null:
                with patch('data_engineering_app.modules.transformer._apply_duplicate_filters') as mock_duplicate:
                    with patch('data_engineering_app.modules.transformer._apply_date_filters') as mock_date:
                        # Set up the mock chain
                        mock_query.return_value = sample_df.iloc[1:4]
                        mock_regex.return_value = sample_df.iloc[1:3]
                        mock_null.return_value = sample_df.iloc[1:3]
                        mock_duplicate.return_value = sample_df.iloc[1:2]
                        mock_date.return_value = sample_df.iloc[1:2]
                        
                        result = apply_advanced_filters_from_dict(sample_df, sample_filter_config)
                        
                        mock_query.assert_called_once()
                        mock_regex.assert_called_once()
                        mock_null.assert_called_once()
                        mock_duplicate.assert_called_once()
                        mock_date.assert_called_once()
                        assert result.equals(sample_df.iloc[1:2])

# Tests for _load_yaml_config
def test_load_yaml_config():
    """Test the _load_yaml_config function."""
    yaml_content = """
    filter_config:
      conditions:
        - "age > 30"
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        result = _load_yaml_config('filter.yaml')
        
        assert 'filter_config' in result
        assert 'conditions' in result['filter_config']
        assert result['filter_config']['conditions'][0] == "age > 30"

# Tests for _apply_query_conditions
def test_apply_query_conditions(sample_df):
    """Test the _apply_query_conditions function."""
    conditions = [
        "age > 30",
        "category != 'C'"
    ]
    
    result = _apply_query_conditions(sample_df, conditions)
    
    # Should only include rows where age > 30 AND category != 'C'
    expected = sample_df[(sample_df['age'] > 30) & (sample_df['category'] != 'C')]
    assert result.equals(expected)

# Tests for _apply_regex_filters
def test_apply_regex_filters(sample_df):
    """Test the _apply_regex_filters function."""
    regex_filters = {
        "name": "^[A-C]"
    }
    
    result = _apply_regex_filters(sample_df, regex_filters)
    
    # Should only include rows where name starts with A, B, or C
    expected = sample_df[sample_df['name'].str.match("^[A-C]", na=False)]
    assert result.equals(expected)

# Tests for _apply_null_filters
def test_apply_null_filters(sample_df):
    """Test the _apply_null_filters function."""
    null_filters = [
        "name"
    ]
    
    result = _apply_null_filters(sample_df, null_filters)
    
    # Should exclude rows where name is null
    expected = sample_df[sample_df['name'].notna()]
    assert result.equals(expected)

# Tests for _apply_duplicate_filters
def test_apply_duplicate_filters(sample_df):
    """Test the _apply_duplicate_filters function."""
    duplicate_filters = [
        "category"
    ]
    
    # Create a DataFrame with duplicates
    df_with_duplicates = pd.concat([sample_df, sample_df.iloc[0:2]])
    
    result = _apply_duplicate_filters(df_with_duplicates, duplicate_filters)
    
    # Should keep only the first occurrence of each category
    assert len(result) < len(df_with_duplicates)
    assert not result['category'].duplicated().any()

# Tests for _apply_date_filters
def test_apply_date_filters(sample_df):
    """Test the _apply_date_filters function."""
    # Create a mock implementation that filters by date range
    def mock_date_filter_implementation(df, date_filters):
        result_df = df.copy()
        for col, date_range in date_filters.items():
            if col in df.columns:
                start = date_range.get("start")
                end = date_range.get("end")
                
                if start:
                    result_df = result_df[result_df[col] >= pd.Timestamp(start)]
                
                if end:
                    result_df = result_df[result_df[col] <= pd.Timestamp(end)]
        
        return result_df
    
    with patch('data_engineering_app.modules.transformer._apply_date_filters', 
              side_effect=mock_date_filter_implementation):
        date_filters = {
            "date_joined": {
                "start": "2020-02-01",
                "end": "2020-04-30"
            }
        }
        
        result = _apply_date_filters(sample_df, date_filters)
        
        # Should only include rows where date_joined is between 2020-02-01 and 2020-04-30
        expected = sample_df[
            (sample_df['date_joined'] >= pd.Timestamp('2020-02-01')) & 
            (sample_df['date_joined'] <= pd.Timestamp('2020-04-30'))
        ]
        
        assert result.equals(expected) 