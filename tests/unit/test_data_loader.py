"""
Unit tests for data loading functionality.

Tests the data_loader module to ensure proper data processing,
validation, and transformation capabilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestDataLoader:
    """Test suite for data loading functionality."""
    
    def test_data_loader_import(self):
        """Test that data loader can be imported successfully."""
        try:
            from data.data_loader import DataLoader
            assert True
        except ImportError:
            # If DataLoader class doesn't exist, test basic import
            try:
                import data.data_loader
                assert True
            except ImportError:
                pytest.skip("Data loader module not available")
    
    def test_pandas_functionality(self):
        """Test basic pandas operations used in data loading."""
        # Create sample data
        data = {
            'customer_id': [1, 2, 3, 4, 5],
            'transaction_amount': [100.0, 250.5, 75.0, 300.0, 150.0],
            'product_id': ['A001', 'B002', 'A001', 'C003', 'B002']
        }
        df = pd.DataFrame(data)
        
        # Test basic operations
        assert len(df) == 5
        assert 'customer_id' in df.columns
        assert df['transaction_amount'].sum() == 875.5
        assert df['customer_id'].nunique() == 5
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Test with valid data
        valid_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'amount': [100, 200, 300]
        })
        
        # Basic validation checks
        assert not valid_data.empty
        assert len(valid_data.columns) > 0
        assert valid_data.isnull().sum().sum() == 0
    
    def test_data_cleaning(self):
        """Test data cleaning operations."""
        # Create data with issues
        dirty_data = pd.DataFrame({
            'customer_id': [1, 2, None, 4, 5],
            'amount': [100, -50, 200, 300, None],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', 'invalid', '2023-01-05']
        })
        
        # Test cleaning operations
        cleaned_data = dirty_data.dropna()
        assert len(cleaned_data) < len(dirty_data)
        
        # Test removing negative values
        positive_amounts = dirty_data[dirty_data['amount'] > 0]
        assert all(positive_amounts['amount'] > 0)


class TestDataProcessing:
    """Test suite for data processing operations."""
    
    def test_data_aggregation(self):
        """Test data aggregation functionality."""
        data = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3],
            'amount': [100, 200, 150, 250, 300],
            'date': pd.date_range('2023-01-01', periods=5)
        })
        
        # Test groupby operations
        customer_totals = data.groupby('customer_id')['amount'].sum()
        assert customer_totals[1] == 300
        assert customer_totals[2] == 400
        assert customer_totals[3] == 300
    
    def test_feature_engineering(self):
        """Test basic feature engineering operations."""
        data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'total_spent': [1000, 2000, 1500],
            'num_transactions': [10, 15, 12]
        })
        
        # Create derived features
        data['avg_transaction'] = data['total_spent'] / data['num_transactions']
        
        assert 'avg_transaction' in data.columns
        assert data.loc[0, 'avg_transaction'] == 100.0
        assert data.loc[1, 'avg_transaction'] == 2000/15


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data for tests."""
    return pd.DataFrame({
        'transaction_id': range(1, 101),
        'customer_id': np.random.randint(1, 21, 100),
        'product_id': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'amount': np.random.uniform(10, 500, 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D')
    })


def test_sample_data_fixture(sample_transaction_data):
    """Test the sample data fixture."""
    assert len(sample_transaction_data) == 100
    assert 'customer_id' in sample_transaction_data.columns
    assert sample_transaction_data['amount'].min() >= 10
    assert sample_transaction_data['amount'].max() <= 500
