# API Documentation

## Core Models

### CustomerSegmentation
Advanced customer segmentation using RFM analysis and clustering.

### ChurnPrediction
ML-based customer churn prediction with retention strategies.

### InventoryOptimization
Demand forecasting and inventory management optimization.

### PricingOptimization
Dynamic pricing strategies and price elasticity analysis.

### FraudDetection
Real-time fraud detection and risk management.

### MarketingAttribution
Multi-touch attribution modeling and ROI analysis.

## Usage Examples

```python
from src.models.customer_segmentation import CustomerSegmentation
from src.models.churn_prediction import ChurnPredictionEngine

# Customer segmentation
segmentation = CustomerSegmentation('data/processed/transactions.csv', 'data/processed/customers.csv')
segments = segmentation.advanced_clustering()

# Churn prediction
churn_engine = ChurnPredictionEngine('data/processed/transactions.csv', 'data/processed/customers.csv')
predictions = churn_engine.predict_churn_risk()
```
