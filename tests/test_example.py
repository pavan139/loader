import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from data_engineering_app.example import main

# Test fixtures
@pytest.fixture
def mock_load_and_audit():
    """Mock the load_and_audit_data function."""
    with patch('data_engineering_app.example.load_and_audit_data') as mock:
        # Create sample DataFrames to return
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })
        
        df2 = pd.DataFrame({
            'id': [1, 2, 4],
            'category': ['A', 'B', 'C'],
            'score': [90, 80, 70]
        })
        
        # Set up the mock to return different DataFrames based on the file path
        def side_effect(file_path, file_type, audit_config):
            if 'sample1.csv' in file_path:
                return df1
            else:
                return df2
        
        mock.side_effect = side_effect
        yield mock

@pytest.fixture
def mock_apply_filters():
    """Mock the apply_advanced_filters_from_config function."""
    with patch('data_engineering_app.example.apply_advanced_filters_from_config') as mock:
        # Return a filtered DataFrame
        filtered_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'value': [100, 200]
        })
        
        mock.return_value = filtered_df
        yield mock

@pytest.fixture
def mock_join_dataframes():
    """Mock the join_dataframes function."""
    with patch('data_engineering_app.example.join_dataframes') as mock:
        # Return a joined DataFrame
        joined_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'value': [100, 200],
            'category': ['A', 'B'],
            'score': [90, 80]
        })
        
        mock.return_value = joined_df
        yield mock

# Test the main function
def test_main(mock_load_and_audit, mock_apply_filters, mock_join_dataframes):
    """Test the main function."""
    with patch('builtins.print') as mock_print:
        # Run the main function
        result = main()
        
        # Check that all the expected functions were called
        assert mock_load_and_audit.call_count == 2
        mock_apply_filters.assert_called_once()
        mock_join_dataframes.assert_called_once()
        
        # Check that the result is the joined DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 5)
        assert list(result.columns) == ['id', 'name', 'value', 'category', 'score']
        
        # Check that print was called to display the results
        mock_print.assert_called()

# Test error handling in main
def test_main_error_handling():
    """Test error handling in the main function."""
    with patch('data_engineering_app.example.load_and_audit_data', side_effect=Exception("Test error")):
        with patch('logging.error') as mock_error:
            with pytest.raises(Exception):
                main()
            
            # Check that the error was logged
            mock_error.assert_called_once()
            assert "Test error" in mock_error.call_args[0][0] 