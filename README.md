# Customer Segmentation & Personalized Recommendation Engine

## Project Overview
This project implements advanced customer segmentation and personalized recommendation systems for retail businesses, with a focus on Norwegian grocery market patterns. The solution addresses key business challenges in customer retention, targeted marketing, and personalized customer experiences.

## Business Problem
- **Customer Segmentation**: Identify distinct customer groups based on purchasing behavior, membership value, and engagement patterns
- **Personalized Recommendations**: Deliver relevant product suggestions to increase basket size and customer satisfaction
- **Targeted Marketing**: Enable data-driven marketing campaigns for different customer segments
- **Customer Value Optimization**: Maximize lifetime value through personalized experiences

## Technical Stack
- **Python**: Core development language
- **SQL/BigQuery**: Data processing and analysis
- **scikit-learn, XGBoost**: Machine learning models
- **Pandas, PySpark**: Data manipulation
- **Docker**: Containerization
- **dbt**: Data transformation
- **Git**: Version control

## Key Features
1. **RFM Analysis** - Recency, Frequency, Monetary segmentation
2. **Collaborative Filtering** - User-based and item-based recommendations
3. **Content-Based Filtering** - Product similarity recommendations
4. **Hybrid Recommendation System** - Combines multiple approaches
5. **A/B Testing Framework** - Measure recommendation effectiveness
6. **Real-time Scoring** - Production-ready model deployment

## Dataset Focus
The project uses four key datasets that represent typical retail data:
- **Transactional Data**: Purchase history, basket composition, seasonal patterns
- **Customer Data**: Demographics, loyalty program status, preferences
- **Product Data**: Categories, attributes, pricing, seasonal trends
- **Digital Events**: Website interactions, email engagement, app usage

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run customer segmentation
python src/customer_segmentation.py

# Generate recommendations
python src/recommendation_engine.py

# Execute full pipeline
python src/main.py
```

## Business Impact
- Increase customer lifetime value by 15-25%
- Improve cross-selling through targeted recommendations
- Reduce marketing costs through precise segmentation
- Enhance customer satisfaction with personalized experiences 