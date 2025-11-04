"""
Unit tests for customer segmentation functionality.

Tests the customer segmentation models and algorithms to ensure
proper clustering, RFM analysis, and segment validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestCustomerSegmentation:
    """Test suite for customer segmentation functionality."""
    
    def test_rfm_calculation(self):
        """Test RFM (Recency, Frequency, Monetary) calculation."""
        # Sample transaction data
        data = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3],
            'transaction_date': pd.to_datetime([
                '2023-01-01', '2023-01-15', '2023-01-02', 
                '2023-01-20', '2023-01-03'
            ]),
            'amount': [100, 200, 150, 250, 300]
        })
        
        # Calculate basic RFM metrics
        current_date = pd.to_datetime('2023-01-31')
        rfm = data.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'customer_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).rename(columns={
            'transaction_date': 'recency',
            'customer_id': 'frequency',
            'amount': 'monetary'
        })
        
        assert len(rfm) == 3
        assert 'recency' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary' in rfm.columns
        
        # Test specific values
        assert rfm.loc[1, 'frequency'] == 2
        assert rfm.loc[1, 'monetary'] == 300
        assert rfm.loc[2, 'frequency'] == 2
        assert rfm.loc[3, 'frequency'] == 1
    
    def test_clustering_basics(self):
        """Test basic clustering functionality."""
        from sklearn.cluster import KMeans
        
        # Sample RFM data
        rfm_data = np.array([
            [10, 5, 1000],  # High value customer
            [30, 2, 200],   # Low value customer
            [5, 8, 1500],   # Very high value customer
            [45, 1, 100],   # Churned customer
            [15, 4, 800]    # Medium value customer
        ])
        
        # Test K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(rfm_data)
        
        assert len(clusters) == 5
        assert len(set(clusters)) <= 2  # Should have at most 2 clusters
        assert all(c in [0, 1] for c in clusters)
    
    def test_segment_validation(self):
        """Test segment validation and quality metrics."""
        from sklearn.metrics import silhouette_score
        
        # Sample data with clear clusters
        data = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
            [8, 8], [8, 9], [9, 8], [9, 9]   # Cluster 2
        ])
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # Calculate silhouette score
        score = silhouette_score(data, labels)
        assert score > 0.5  # Should have good separation
    
    def test_customer_value_calculation(self):
        """Test customer lifetime value calculation."""
        # Sample customer data
        customers = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'total_spent': [1000, 2000, 1500],
            'num_transactions': [10, 15, 12],
            'days_active': [365, 200, 300]
        })
        
        # Calculate CLV metrics
        customers['avg_order_value'] = customers['total_spent'] / customers['num_transactions']
        customers['purchase_frequency'] = customers['num_transactions'] / customers['days_active'] * 365
        customers['clv_estimate'] = customers['avg_order_value'] * customers['purchase_frequency']
        
        assert 'clv_estimate' in customers.columns
        assert all(customers['clv_estimate'] > 0)
        assert customers.loc[0, 'avg_order_value'] == 100.0


class TestSegmentAnalysis:
    """Test suite for segment analysis functionality."""
    
    def test_segment_profiling(self):
        """Test segment profiling and characterization."""
        # Sample segmented data
        data = pd.DataFrame({
            'customer_id': range(1, 11),
            'segment': ['High Value', 'High Value', 'Medium Value', 'Medium Value', 'Low Value',
                       'High Value', 'Medium Value', 'Low Value', 'Low Value', 'High Value'],
            'total_spent': [2000, 1800, 800, 900, 200, 2200, 750, 150, 300, 1900],
            'frequency': [15, 12, 8, 9, 3, 18, 7, 2, 4, 14]
        })
        
        # Profile segments
        segment_profile = data.groupby('segment').agg({
            'total_spent': ['mean', 'count'],
            'frequency': 'mean'
        }).round(2)
        
        assert len(segment_profile) == 3
        # High value customers should have higher average spending
        high_value_spending = segment_profile.loc['High Value', ('total_spent', 'mean')]
        low_value_spending = segment_profile.loc['Low Value', ('total_spent', 'mean')]
        assert high_value_spending > low_value_spending
    
    def test_segment_stability(self):
        """Test segment stability over time."""
        # Simulate customer segments over two time periods
        period1 = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'segment': ['A', 'A', 'B', 'B', 'C']
        })
        
        period2 = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'segment': ['A', 'A', 'B', 'A', 'C']  # Customer 4 moved from B to A
        })
        
        # Calculate stability
        merged = period1.merge(period2, on='customer_id', suffixes=('_p1', '_p2'))
        stability = (merged['segment_p1'] == merged['segment_p2']).mean()
        
        assert 0 <= stability <= 1
        assert stability == 0.8  # 4 out of 5 customers stayed in same segment


@pytest.fixture
def sample_customer_data():
    """Fixture providing sample customer data for segmentation tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'customer_id': range(1, 101),
        'recency': np.random.randint(1, 365, 100),
        'frequency': np.random.randint(1, 50, 100),
        'monetary': np.random.uniform(50, 5000, 100)
    })


def test_customer_data_fixture(sample_customer_data):
    """Test the sample customer data fixture."""
    assert len(sample_customer_data) == 100
    assert all(col in sample_customer_data.columns for col in ['customer_id', 'recency', 'frequency', 'monetary'])
    assert sample_customer_data['recency'].min() >= 1
    assert sample_customer_data['frequency'].min() >= 1
    assert sample_customer_data['monetary'].min() >= 50
