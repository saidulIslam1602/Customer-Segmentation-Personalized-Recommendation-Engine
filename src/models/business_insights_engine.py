"""
Business Insights Engine - Automated Insight Generation and Anomaly Detection
Provides intelligent business insights, KPI monitoring, and executive summaries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

class BusinessInsightsEngine:
    """
    Automated Business Insights and Intelligence Engine
    
    Features:
    - Automated insight generation from data patterns
    - KPI anomaly detection and alerting
    - Executive summary generation
    - Trend analysis and forecasting
    - Performance benchmarking
    - Action item recommendations
    - Business health scoring
    """
    
    def __init__(self, transactions_path, customers_path, products_path, results_dir='reports/insights'):
        """Initialize the business insights engine"""
        self.transactions = pd.read_csv(transactions_path)
        self.customers = pd.read_csv(customers_path)
        self.products = pd.read_csv(products_path)
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Insights storage
        self.insights = []
        self.anomalies = []
        self.kpis = {}
        self.trends = {}
        self.recommendations = []
        
        # Business rules and thresholds
        self.thresholds = {
            'revenue_decline_threshold': -0.05,  # 5% decline
            'customer_churn_threshold': 0.15,    # 15% churn rate
            'inventory_turnover_min': 4,         # Minimum turns per year
            'profit_margin_min': 0.15,          # 15% minimum margin
            'anomaly_sensitivity': 0.05          # 5% contamination
        }
        
    def calculate_business_kpis(self):
        """Calculate comprehensive business KPIs"""
        print("📊 Calculating business KPIs...")
        
        # Prepare transaction data
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        current_date = self.transactions['transaction_date'].max()
        
        # Revenue KPIs
        total_revenue = self.transactions['total_amount'].sum()
        avg_order_value = self.transactions['total_amount'].mean()
        
        # Customer KPIs
        total_customers = self.transactions['customer_id'].nunique()
        avg_customer_value = total_revenue / total_customers
        
        # Product KPIs
        total_products_sold = self.transactions['quantity'].sum()
        avg_items_per_order = self.transactions.groupby('transaction_id')['quantity'].sum().mean()
        
        # Time-based KPIs
        date_range = (current_date - self.transactions['transaction_date'].min()).days
        daily_revenue = total_revenue / date_range if date_range > 0 else 0
        
        # Monthly trends
        monthly_data = self.transactions.groupby(
            self.transactions['transaction_date'].dt.to_period('M')
        ).agg({
            'total_amount': 'sum',
            'customer_id': 'nunique',
            'transaction_id': 'count'
        }).reset_index()
        
        # Growth rates
        if len(monthly_data) >= 2:
            revenue_growth = (monthly_data['total_amount'].iloc[-1] - monthly_data['total_amount'].iloc[-2]) / monthly_data['total_amount'].iloc[-2] * 100
            customer_growth = (monthly_data['customer_id'].iloc[-1] - monthly_data['customer_id'].iloc[-2]) / monthly_data['customer_id'].iloc[-2] * 100
        else:
            revenue_growth = 0
            customer_growth = 0
        
        # Customer behavior KPIs
        customer_frequency = self.transactions.groupby('customer_id').size()
        repeat_customer_rate = (customer_frequency > 1).mean() * 100
        
        # Product performance
        product_performance = self.transactions.groupby('product_id').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        top_products = product_performance.nlargest(10, 'total_amount')
        
        self.kpis = {
            'financial': {
                'total_revenue': total_revenue,
                'avg_order_value': avg_order_value,
                'daily_revenue': daily_revenue,
                'revenue_growth_rate': revenue_growth
            },
            'customer': {
                'total_customers': total_customers,
                'avg_customer_value': avg_customer_value,
                'repeat_customer_rate': repeat_customer_rate,
                'customer_growth_rate': customer_growth
            },
            'operational': {
                'total_products_sold': total_products_sold,
                'avg_items_per_order': avg_items_per_order,
                'total_transactions': len(self.transactions),
                'date_range_days': date_range
            },
            'product': {
                'unique_products': self.transactions['product_id'].nunique(),
                'top_products': top_products.to_dict('records')
            }
        }
        
        print(f"✅ KPIs calculated")
        print(f"   💰 Total Revenue: ${total_revenue:,.2f}")
        print(f"   👥 Total Customers: {total_customers:,}")
        print(f"   📈 Revenue Growth: {revenue_growth:.1f}%")
        print(f"   🔄 Repeat Customer Rate: {repeat_customer_rate:.1f}%")
        
        return self.kpis
    
    def detect_anomalies(self):
        """Detect anomalies in business metrics"""
        print("🔍 Detecting business anomalies...")
        
        if not self.kpis:
            self.calculate_business_kpis()
        
        # Daily revenue anomalies
        daily_revenue = self.transactions.groupby(
            self.transactions['transaction_date'].dt.date
        )['total_amount'].sum().reset_index()
        
        if len(daily_revenue) > 7:  # Need at least a week of data
            # Use Isolation Forest for anomaly detection
            revenue_values = daily_revenue['total_amount'].values.reshape(-1, 1)
            
            iso_forest = IsolationForest(
                contamination=self.thresholds['anomaly_sensitivity'],
                random_state=42
            )
            
            anomaly_labels = iso_forest.fit_predict(revenue_values)
            anomaly_scores = iso_forest.score_samples(revenue_values)
            
            # Identify anomalous days
            anomalous_days = daily_revenue[anomaly_labels == -1].copy()
            anomalous_days['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
            
            for _, row in anomalous_days.iterrows():
                self.anomalies.append({
                    'type': 'revenue_anomaly',
                    'date': row['transaction_date'],
                    'value': row['total_amount'],
                    'severity': 'high' if row['anomaly_score'] < -0.5 else 'medium',
                    'description': f"Unusual revenue of ${row['total_amount']:,.2f} detected"
                })
        
        # Customer behavior anomalies
        customer_daily_counts = self.transactions.groupby(
            self.transactions['transaction_date'].dt.date
        )['customer_id'].nunique().reset_index()
        
        if len(customer_daily_counts) > 7:
            customer_values = customer_daily_counts['customer_id'].values.reshape(-1, 1)
            
            iso_forest_customers = IsolationForest(
                contamination=self.thresholds['anomaly_sensitivity'],
                random_state=42
            )
            
            customer_anomaly_labels = iso_forest_customers.fit_predict(customer_values)
            
            anomalous_customer_days = customer_daily_counts[customer_anomaly_labels == -1]
            
            for _, row in anomalous_customer_days.iterrows():
                self.anomalies.append({
                    'type': 'customer_anomaly',
                    'date': row['transaction_date'],
                    'value': row['customer_id'],
                    'severity': 'medium',
                    'description': f"Unusual customer activity: {row['customer_id']} unique customers"
                })
        
        # Product performance anomalies
        product_sales = self.transactions.groupby('product_id').agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Detect products with unusual sales patterns
        revenue_threshold = product_sales['total_amount'].quantile(0.95)
        top_performers = product_sales[product_sales['total_amount'] > revenue_threshold]
        
        for _, row in top_performers.iterrows():
            product_name = self.products[self.products['product_id'] == row['product_id']]['product_name'].iloc[0] if len(self.products[self.products['product_id'] == row['product_id']]) > 0 else 'Unknown'
            
            self.anomalies.append({
                'type': 'product_performance',
                'product_id': row['product_id'],
                'product_name': product_name,
                'value': row['total_amount'],
                'severity': 'positive',
                'description': f"Exceptional performance: {product_name} generated ${row['total_amount']:,.2f}"
            })
        
        print(f"✅ Anomaly detection completed")
        print(f"   🚨 Total anomalies detected: {len(self.anomalies)}")
        
        # Print summary by type
        anomaly_types = {}
        for anomaly in self.anomalies:
            anomaly_type = anomaly['type']
            anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
        
        for anom_type, count in anomaly_types.items():
            print(f"   📊 {anom_type}: {count}")
        
        return self.anomalies
    
    def analyze_trends(self):
        """Analyze business trends and patterns"""
        print("📈 Analyzing business trends...")
        
        # Monthly trends
        monthly_trends = self.transactions.groupby(
            self.transactions['transaction_date'].dt.to_period('M')
        ).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        monthly_trends.columns = ['month', 'revenue_total', 'revenue_avg', 'transaction_count', 'unique_customers', 'unique_products']
        
        # Calculate trends
        if len(monthly_trends) >= 3:
            # Revenue trend
            revenue_trend = np.polyfit(range(len(monthly_trends)), monthly_trends['revenue_total'], 1)[0]
            
            # Customer trend
            customer_trend = np.polyfit(range(len(monthly_trends)), monthly_trends['unique_customers'], 1)[0]
            
            # Transaction trend
            transaction_trend = np.polyfit(range(len(monthly_trends)), monthly_trends['transaction_count'], 1)[0]
            
            self.trends = {
                'revenue_trend': {
                    'slope': revenue_trend,
                    'direction': 'increasing' if revenue_trend > 0 else 'decreasing',
                    'monthly_data': monthly_trends[['month', 'revenue_total']].to_dict('records')
                },
                'customer_trend': {
                    'slope': customer_trend,
                    'direction': 'increasing' if customer_trend > 0 else 'decreasing',
                    'monthly_data': monthly_trends[['month', 'unique_customers']].to_dict('records')
                },
                'transaction_trend': {
                    'slope': transaction_trend,
                    'direction': 'increasing' if transaction_trend > 0 else 'decreasing',
                    'monthly_data': monthly_trends[['month', 'transaction_count']].to_dict('records')
                }
            }
        
        # Seasonal patterns
        seasonal_data = self.transactions.copy()
        seasonal_data['month'] = seasonal_data['transaction_date'].dt.month
        seasonal_data['day_of_week'] = seasonal_data['transaction_date'].dt.dayofweek
        
        monthly_seasonality = seasonal_data.groupby('month')['total_amount'].sum()
        weekly_seasonality = seasonal_data.groupby('day_of_week')['total_amount'].sum()
        
        self.trends['seasonality'] = {
            'monthly': monthly_seasonality.to_dict(),
            'weekly': weekly_seasonality.to_dict(),
            'peak_month': monthly_seasonality.idxmax(),
            'peak_day': weekly_seasonality.idxmax()
        }
        
        print(f"✅ Trend analysis completed")
        if self.trends:
            print(f"   📈 Revenue trend: {self.trends['revenue_trend']['direction']}")
            print(f"   👥 Customer trend: {self.trends['customer_trend']['direction']}")
            print(f"   🗓️ Peak month: {self.trends['seasonality']['peak_month']}")
        
        return self.trends
    
    def generate_insights(self):
        """Generate automated business insights"""
        print("💡 Generating business insights...")
        
        if not self.kpis:
            self.calculate_business_kpis()
        
        if not self.trends:
            self.analyze_trends()
        
        # Revenue insights
        revenue_growth = self.kpis['financial']['revenue_growth_rate']
        if revenue_growth > 10:
            self.insights.append({
                'category': 'financial',
                'type': 'positive',
                'title': 'Strong Revenue Growth',
                'description': f"Revenue is growing at {revenue_growth:.1f}% month-over-month, indicating strong business performance.",
                'impact': 'high',
                'confidence': 0.9
            })
        elif revenue_growth < -5:
            self.insights.append({
                'category': 'financial',
                'type': 'warning',
                'title': 'Revenue Decline Alert',
                'description': f"Revenue has declined by {abs(revenue_growth):.1f}% month-over-month. Immediate attention required.",
                'impact': 'high',
                'confidence': 0.95
            })
        
        # Customer insights
        repeat_rate = self.kpis['customer']['repeat_customer_rate']
        if repeat_rate > 60:
            self.insights.append({
                'category': 'customer',
                'type': 'positive',
                'title': 'High Customer Loyalty',
                'description': f"Excellent repeat customer rate of {repeat_rate:.1f}% indicates strong customer loyalty and satisfaction.",
                'impact': 'medium',
                'confidence': 0.85
            })
        elif repeat_rate < 30:
            self.insights.append({
                'category': 'customer',
                'type': 'warning',
                'title': 'Low Customer Retention',
                'description': f"Repeat customer rate of {repeat_rate:.1f}% is below industry standards. Focus on retention strategies.",
                'impact': 'high',
                'confidence': 0.9
            })
        
        # Product insights
        avg_items = self.kpis['operational']['avg_items_per_order']
        if avg_items > 3:
            self.insights.append({
                'category': 'product',
                'type': 'positive',
                'title': 'Strong Cross-Selling Performance',
                'description': f"Average of {avg_items:.1f} items per order suggests effective cross-selling strategies.",
                'impact': 'medium',
                'confidence': 0.8
            })
        
        # Seasonal insights
        if 'seasonality' in self.trends:
            peak_month = self.trends['seasonality']['peak_month']
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            
            self.insights.append({
                'category': 'operational',
                'type': 'informational',
                'title': 'Seasonal Pattern Identified',
                'description': f"Peak sales occur in {month_names.get(peak_month, 'Unknown')}. Plan inventory and marketing accordingly.",
                'impact': 'medium',
                'confidence': 0.75
            })
        
        # Anomaly-based insights
        high_severity_anomalies = [a for a in self.anomalies if a.get('severity') == 'high']
        if high_severity_anomalies:
            self.insights.append({
                'category': 'operational',
                'type': 'alert',
                'title': 'Business Anomalies Detected',
                'description': f"{len(high_severity_anomalies)} high-severity anomalies detected in recent business metrics.",
                'impact': 'high',
                'confidence': 0.9
            })
        
        print(f"✅ Generated {len(self.insights)} business insights")
        
        # Print insight summary
        insight_categories = {}
        for insight in self.insights:
            category = insight['category']
            insight_categories[category] = insight_categories.get(category, 0) + 1
        
        for category, count in insight_categories.items():
            print(f"   📊 {category}: {count} insights")
        
        return self.insights
    
    def generate_recommendations(self):
        """Generate actionable business recommendations"""
        print("🎯 Generating business recommendations...")
        
        # Revenue optimization recommendations
        if self.kpis['financial']['revenue_growth_rate'] < 0:
            self.recommendations.append({
                'category': 'revenue',
                'priority': 'high',
                'title': 'Implement Revenue Recovery Plan',
                'actions': [
                    'Launch targeted marketing campaigns',
                    'Offer limited-time promotions',
                    'Analyze competitor pricing',
                    'Improve customer retention programs'
                ],
                'expected_impact': 'Potential 10-15% revenue increase',
                'timeline': '30-60 days'
            })
        
        # Customer retention recommendations
        if self.kpis['customer']['repeat_customer_rate'] < 40:
            self.recommendations.append({
                'category': 'customer',
                'priority': 'high',
                'title': 'Enhance Customer Retention',
                'actions': [
                    'Implement loyalty program',
                    'Personalize customer communications',
                    'Improve customer service quality',
                    'Create customer feedback loops'
                ],
                'expected_impact': 'Increase repeat rate by 15-20%',
                'timeline': '60-90 days'
            })
        
        # Product optimization recommendations
        top_products = self.kpis['product']['top_products'][:3]
        if top_products:
            self.recommendations.append({
                'category': 'product',
                'priority': 'medium',
                'title': 'Optimize Product Mix',
                'actions': [
                    f'Increase inventory for top performers',
                    'Bundle high-performing products',
                    'Analyze underperforming products',
                    'Implement dynamic pricing'
                ],
                'expected_impact': 'Improve profit margins by 5-10%',
                'timeline': '30-45 days'
            })
        
        # Operational efficiency recommendations
        if len(self.anomalies) > 5:
            self.recommendations.append({
                'category': 'operational',
                'priority': 'medium',
                'title': 'Improve Operational Consistency',
                'actions': [
                    'Implement real-time monitoring',
                    'Standardize business processes',
                    'Train staff on best practices',
                    'Set up automated alerts'
                ],
                'expected_impact': 'Reduce operational variance by 20%',
                'timeline': '45-60 days'
            })
        
        print(f"✅ Generated {len(self.recommendations)} recommendations")
        
        for rec in self.recommendations:
            print(f"   🎯 {rec['title']} ({rec['priority']} priority)")
        
        return self.recommendations
    
    def calculate_business_health_score(self):
        """Calculate overall business health score"""
        print("🏥 Calculating business health score...")
        
        if not self.kpis:
            self.calculate_business_kpis()
        
        # Scoring components (0-100 scale)
        scores = {}
        
        # Revenue health (30% weight)
        revenue_growth = self.kpis['financial']['revenue_growth_rate']
        if revenue_growth > 15:
            scores['revenue'] = 100
        elif revenue_growth > 5:
            scores['revenue'] = 80
        elif revenue_growth > 0:
            scores['revenue'] = 60
        elif revenue_growth > -5:
            scores['revenue'] = 40
        else:
            scores['revenue'] = 20
        
        # Customer health (25% weight)
        repeat_rate = self.kpis['customer']['repeat_customer_rate']
        if repeat_rate > 70:
            scores['customer'] = 100
        elif repeat_rate > 50:
            scores['customer'] = 80
        elif repeat_rate > 30:
            scores['customer'] = 60
        elif repeat_rate > 15:
            scores['customer'] = 40
        else:
            scores['customer'] = 20
        
        # Operational health (25% weight)
        anomaly_count = len([a for a in self.anomalies if a.get('severity') in ['high', 'medium']])
        if anomaly_count == 0:
            scores['operational'] = 100
        elif anomaly_count <= 2:
            scores['operational'] = 80
        elif anomaly_count <= 5:
            scores['operational'] = 60
        elif anomaly_count <= 10:
            scores['operational'] = 40
        else:
            scores['operational'] = 20
        
        # Growth health (20% weight)
        customer_growth = self.kpis['customer']['customer_growth_rate']
        if customer_growth > 10:
            scores['growth'] = 100
        elif customer_growth > 5:
            scores['growth'] = 80
        elif customer_growth > 0:
            scores['growth'] = 60
        elif customer_growth > -5:
            scores['growth'] = 40
        else:
            scores['growth'] = 20
        
        # Calculate weighted score
        weights = {'revenue': 0.30, 'customer': 0.25, 'operational': 0.25, 'growth': 0.20}
        health_score = sum(scores[component] * weights[component] for component in scores)
        
        # Health status
        if health_score >= 80:
            status = 'Excellent'
        elif health_score >= 60:
            status = 'Good'
        elif health_score >= 40:
            status = 'Fair'
        else:
            status = 'Poor'
        
        health_report = {
            'overall_score': health_score,
            'status': status,
            'component_scores': scores,
            'weights': weights,
            'calculation_date': datetime.now().isoformat()
        }
        
        print(f"✅ Business health score calculated: {health_score:.1f}/100 ({status})")
        print(f"   💰 Revenue Health: {scores['revenue']}/100")
        print(f"   👥 Customer Health: {scores['customer']}/100")
        print(f"   ⚙️  Operational Health: {scores['operational']}/100")
        print(f"   📈 Growth Health: {scores['growth']}/100")
        
        return health_report
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        print("📋 Generating executive summary...")
        
        # Ensure all analyses are complete
        if not self.kpis:
            self.calculate_business_kpis()
        if not self.insights:
            self.generate_insights()
        if not self.recommendations:
            self.generate_recommendations()
        
        health_score = self.calculate_business_health_score()
        
        # Create executive summary
        summary = {
            'report_date': datetime.now().isoformat(),
            'business_health': health_score,
            'key_metrics': {
                'total_revenue': self.kpis['financial']['total_revenue'],
                'revenue_growth': self.kpis['financial']['revenue_growth_rate'],
                'total_customers': self.kpis['customer']['total_customers'],
                'customer_growth': self.kpis['customer']['customer_growth_rate'],
                'repeat_customer_rate': self.kpis['customer']['repeat_customer_rate']
            },
            'top_insights': [insight for insight in self.insights if insight['impact'] == 'high'][:3],
            'critical_recommendations': [rec for rec in self.recommendations if rec['priority'] == 'high'],
            'anomalies_summary': {
                'total_anomalies': len(self.anomalies),
                'high_severity': len([a for a in self.anomalies if a.get('severity') == 'high']),
                'recent_anomalies': [a for a in self.anomalies if a.get('severity') == 'high'][:3]
            }
        }
        
        print(f"✅ Executive summary generated")
        print(f"   🏥 Business Health: {health_score['overall_score']:.1f}/100 ({health_score['status']})")
        print(f"   💡 Key Insights: {len(summary['top_insights'])}")
        print(f"   🎯 Critical Recommendations: {len(summary['critical_recommendations'])}")
        
        return summary
    
    def run_complete_analysis(self):
        """Run complete business intelligence analysis"""
        print("🚀 RUNNING COMPLETE BUSINESS INTELLIGENCE ANALYSIS")
        print("=" * 70)
        
        # Step 1: Calculate KPIs
        self.calculate_business_kpis()
        
        # Step 2: Detect anomalies
        self.detect_anomalies()
        
        # Step 3: Analyze trends
        self.analyze_trends()
        
        # Step 4: Generate insights
        self.generate_insights()
        
        # Step 5: Generate recommendations
        self.generate_recommendations()
        
        # Step 6: Create executive summary
        executive_summary = self.generate_executive_summary()
        
        # Step 7: Save results
        self.save_results(executive_summary)
        
        print(f"\n🎉 BUSINESS INTELLIGENCE ANALYSIS COMPLETED!")
        
        return {
            'kpis': self.kpis,
            'insights': self.insights,
            'anomalies': self.anomalies,
            'trends': self.trends,
            'recommendations': self.recommendations,
            'executive_summary': executive_summary
        }
    
    def save_results(self, executive_summary):
        """Save all analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save executive summary
        with open(f'{self.results_dir}/executive_summary_{timestamp}.json', 'w') as f:
            json.dump(executive_summary, f, indent=2, default=str)
        
        # Save detailed results
        detailed_results = {
            'timestamp': timestamp,
            'kpis': self.kpis,
            'insights': self.insights,
            'anomalies': self.anomalies,
            'trends': self.trends,
            'recommendations': self.recommendations
        }
        
        with open(f'{self.results_dir}/detailed_analysis_{timestamp}.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"💾 Results saved with timestamp: {timestamp}")
        
        return timestamp

def main():
    """Demo of business insights engine"""
    print("💡 BUSINESS INSIGHTS ENGINE DEMO")
    print("=" * 50)
    
    # Initialize engine
    insights_engine = BusinessInsightsEngine(
        'data/processed/transactions.csv',
        'data/processed/customers.csv',
        'data/processed/products.csv'
    )
    
    # Run complete analysis
    results = insights_engine.run_complete_analysis()
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   KPIs Calculated: ✅")
    print(f"   Insights Generated: {len(results['insights'])}")
    print(f"   Anomalies Detected: {len(results['anomalies'])}")
    print(f"   Recommendations: {len(results['recommendations'])}")
    print(f"   Business Health: {results['executive_summary']['business_health']['overall_score']:.1f}/100")
    
    print(f"\n✅ Business Insights Engine Demo Completed!")

if __name__ == "__main__":
    main()