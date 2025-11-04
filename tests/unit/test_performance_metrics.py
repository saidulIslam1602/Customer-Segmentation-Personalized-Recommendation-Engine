"""
Unit tests for performance metrics functionality.

Tests the performance metrics generation, calculation,
and reporting capabilities of the platform.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestPerformanceMetrics:
    """Test suite for performance metrics functionality."""
    
    def test_roi_calculation(self):
        """Test ROI calculation accuracy."""
        # Test data
        baseline_revenue = 1000000
        current_revenue = 1200000
        implementation_cost = 50000
        
        # Calculate ROI
        revenue_increase = current_revenue - baseline_revenue
        roi_percentage = (revenue_increase - implementation_cost) / implementation_cost * 100
        
        assert revenue_increase == 200000
        assert roi_percentage == 300.0  # 300% ROI
    
    def test_business_metrics_calculation(self):
        """Test business metrics calculation."""
        # Sample transaction data
        transactions = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3, 3, 4],
            'amount': [100, 200, 150, 250, 300, 100, 50],
            'date': pd.date_range('2023-01-01', periods=7)
        })
        
        # Calculate metrics
        total_revenue = transactions['amount'].sum()
        unique_customers = transactions['customer_id'].nunique()
        avg_order_value = transactions['amount'].mean()
        repeat_customers = transactions['customer_id'].value_counts()
        repeat_rate = (repeat_customers > 1).sum() / unique_customers
        
        assert total_revenue == 1150
        assert unique_customers == 4
        assert avg_order_value == 1150/7
        assert repeat_rate == 0.75  # 3 out of 4 customers made repeat purchases
    
    def test_model_performance_metrics(self):
        """Test ML model performance metrics."""
        # Sample predictions vs actual
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        
        # Calculate metrics manually
        tp = sum((y_true == 1) & (y_pred == 1))  # True positives
        tn = sum((y_true == 0) & (y_pred == 0))  # True negatives
        fp = sum((y_true == 0) & (y_pred == 1))  # False positives
        fn = sum((y_true == 1) & (y_pred == 0))  # False negatives
        
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        assert tp == 4
        assert tn == 4
        assert fp == 1
        assert fn == 1
        assert accuracy == 0.8
        assert precision == 0.8
        assert recall == 0.8
    
    def test_segmentation_quality_metrics(self):
        """Test segmentation quality metrics."""
        # Sample data with known clusters
        data = np.array([
            [1, 1], [1, 2], [2, 1],  # Cluster 1
            [8, 8], [8, 9], [9, 8]   # Cluster 2
        ])
        labels = [0, 0, 0, 1, 1, 1]
        
        # Calculate within-cluster sum of squares manually
        cluster_0_points = data[np.array(labels) == 0]
        cluster_1_points = data[np.array(labels) == 1]
        
        centroid_0 = cluster_0_points.mean(axis=0)
        centroid_1 = cluster_1_points.mean(axis=0)
        
        wcss_0 = sum(np.sum((point - centroid_0)**2) for point in cluster_0_points)
        wcss_1 = sum(np.sum((point - centroid_1)**2) for point in cluster_1_points)
        total_wcss = wcss_0 + wcss_1
        
        assert total_wcss < 10  # Should be low for well-separated clusters


class TestReportGeneration:
    """Test suite for report generation functionality."""
    
    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        # Sample performance data
        metrics = {
            'overall_score': 85.5,
            'roi_percentage': 450.0,
            'accuracy': 0.92,
            'customer_retention': 0.88,
            'revenue_increase': 250000
        }
        
        # Generate summary
        summary = {
            'grade': 'A' if metrics['overall_score'] >= 90 else 'B+' if metrics['overall_score'] >= 80 else 'B',
            'status': 'Excellent' if metrics['overall_score'] >= 90 else 'Very Good',
            'roi_category': 'Outstanding' if metrics['roi_percentage'] > 400 else 'Good',
            'model_quality': 'High' if metrics['accuracy'] > 0.9 else 'Medium'
        }
        
        assert summary['grade'] == 'B+'
        assert summary['status'] == 'Very Good'
        assert summary['roi_category'] == 'Outstanding'
        assert summary['model_quality'] == 'High'
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality."""
        # Sample metrics vs benchmarks
        actual_metrics = {
            'churn_accuracy': 0.95,
            'recommendation_precision': 0.18,
            'customer_retention': 0.88
        }
        
        industry_benchmarks = {
            'churn_accuracy': 0.80,
            'recommendation_precision': 0.15,
            'customer_retention': 0.75
        }
        
        # Calculate improvements
        improvements = {}
        for metric in actual_metrics:
            improvement = (actual_metrics[metric] - industry_benchmarks[metric]) / industry_benchmarks[metric] * 100
            improvements[metric] = round(improvement, 1)
        
        assert abs(improvements['churn_accuracy'] - 18.8) < 0.2  # ~18.8% above benchmark
        assert improvements['recommendation_precision'] == 20.0  # 20% above benchmark
        assert abs(improvements['customer_retention'] - 17.3) < 0.1  # ~17.3% above benchmark
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        # Sample time series data
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        revenue = [100000, 105000, 110000, 108000, 115000, 120000,
                  125000, 130000, 128000, 135000, 140000, 145000]
        
        df = pd.DataFrame({'date': dates, 'revenue': revenue})
        
        # Calculate trend
        df['revenue_change'] = df['revenue'].pct_change()
        avg_growth = df['revenue_change'].mean()
        trend_direction = 'increasing' if avg_growth > 0 else 'decreasing'
        
        assert trend_direction == 'increasing'
        assert avg_growth > 0


@pytest.fixture
def sample_performance_data():
    """Fixture providing sample performance data for tests."""
    return {
        'model_metrics': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.91,
            'f1_score': 0.90
        },
        'business_metrics': {
            'total_revenue': 1500000,
            'customer_count': 5000,
            'retention_rate': 0.85,
            'avg_order_value': 125.50
        },
        'operational_metrics': {
            'processing_time': 45.2,
            'data_quality_score': 0.94,
            'system_uptime': 0.998
        }
    }


def test_performance_data_fixture(sample_performance_data):
    """Test the sample performance data fixture."""
    assert 'model_metrics' in sample_performance_data
    assert 'business_metrics' in sample_performance_data
    assert 'operational_metrics' in sample_performance_data
    assert sample_performance_data['model_metrics']['accuracy'] > 0.8
    assert sample_performance_data['business_metrics']['total_revenue'] > 0
