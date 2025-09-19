"""
Advanced Marketing Attribution and ROI Analysis Engine
Addresses critical business issue: Marketing effectiveness and budget optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MarketingAttributionEngine:
    """
    Advanced Marketing Attribution System for Retail Business
    
    Business Problems Addressed:
    - Marketing channel attribution and ROI analysis
    - Customer acquisition cost optimization
    - Campaign effectiveness measurement
    - Budget allocation optimization
    - Multi-touch attribution modeling
    """
    
    def __init__(self, transactions_path, customers_path, digital_events_path):
        """Initialize with transaction, customer, and digital events data"""
        self.transactions = pd.read_csv(transactions_path)
        self.customers = pd.read_csv(customers_path)
        self.digital_events = pd.read_csv(digital_events_path)
        self.attribution_data = None
        self.attribution_model = None
        self.channel_performance = None
        
    def prepare_attribution_data(self):
        """Prepare comprehensive marketing attribution data"""
        print("🔄 Preparing marketing attribution data...")
        
        # Convert dates
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        self.customers['member_since'] = pd.to_datetime(self.customers['member_since'])
        self.digital_events['event_date'] = pd.to_datetime(self.digital_events['event_date'])
        
        # Create customer journey data
        customer_journeys = []
        
        for customer_id in self.customers['customer_id'].unique():
            customer_info = self.customers[self.customers['customer_id'] == customer_id].iloc[0]
            customer_transactions = self.transactions[self.transactions['customer_id'] == customer_id].sort_values('transaction_date')
            customer_events = self.digital_events[self.digital_events['customer_id'] == customer_id].sort_values('event_date')
            
            if len(customer_transactions) == 0:
                continue
            
            # First purchase (acquisition)
            first_purchase = customer_transactions.iloc[0]
            
            # Events before first purchase (attribution window: 30 days)
            attribution_window_start = first_purchase['transaction_date'] - timedelta(days=30)
            pre_purchase_events = customer_events[
                (customer_events['event_date'] >= attribution_window_start) &
                (customer_events['event_date'] <= first_purchase['transaction_date'])
            ]
            
            # Channel touchpoints
            touchpoints = []
            
            # Add digital touchpoints
            for _, event in pre_purchase_events.iterrows():
                touchpoints.append({
                    'channel': f"Digital_{event['event_type']}",
                    'date': event['event_date'],
                    'days_before_purchase': (first_purchase['transaction_date'] - event['event_date']).days
                })
            
            # Add transaction channel as final touchpoint
            touchpoints.append({
                'channel': first_purchase['channel'],
                'date': first_purchase['transaction_date'],
                'days_before_purchase': 0
            })
            
            # Calculate customer metrics
            total_spent = customer_transactions['total_amount'].sum()
            total_transactions = len(customer_transactions)
            customer_lifetime_days = (customer_transactions['transaction_date'].max() - customer_transactions['transaction_date'].min()).days + 1
            
            # Attribution analysis
            if len(touchpoints) > 0:
                # First-touch attribution
                first_touch_channel = touchpoints[0]['channel']
                
                # Last-touch attribution
                last_touch_channel = touchpoints[-1]['channel']
                
                # Multi-touch attribution (time decay)
                channel_weights = {}
                total_weight = 0
                
                for i, touchpoint in enumerate(touchpoints):
                    # Time decay: more recent touchpoints get higher weight
                    weight = 0.5 ** touchpoint['days_before_purchase']
                    channel = touchpoint['channel']
                    
                    if channel not in channel_weights:
                        channel_weights[channel] = 0
                    channel_weights[channel] += weight
                    total_weight += weight
                
                # Normalize weights
                if total_weight > 0:
                    for channel in channel_weights:
                        channel_weights[channel] /= total_weight
                
                customer_journeys.append({
                    'customer_id': customer_id,
                    'first_purchase_date': first_purchase['transaction_date'],
                    'first_purchase_amount': first_purchase['total_amount'],
                    'total_customer_value': total_spent,
                    'total_transactions': total_transactions,
                    'customer_lifetime_days': customer_lifetime_days,
                    'touchpoints_count': len(touchpoints),
                    'first_touch_channel': first_touch_channel,
                    'last_touch_channel': last_touch_channel,
                    'multi_touch_attribution': channel_weights,
                    'age': customer_info['age'],
                    'gender': customer_info['gender'],
                    'loyalty_tier': customer_info['loyalty_tier'],
                    'estimated_income': customer_info['estimated_income'],
                    'household_size': customer_info['household_size'],
                    'email_consent': customer_info['email_consent'],
                    'mobile_app_user': customer_info['mobile_app_user']
                })
        
        attribution_df = pd.DataFrame(customer_journeys)
        
        # Expand multi-touch attribution into separate columns
        all_channels = set()
        for attribution in attribution_df['multi_touch_attribution']:
            all_channels.update(attribution.keys())
        
        for channel in all_channels:
            attribution_df[f'attribution_{channel}'] = attribution_df['multi_touch_attribution'].apply(
                lambda x: x.get(channel, 0)
            )
        
        # Calculate time-based features
        attribution_df['days_since_first_purchase'] = (
            attribution_df['first_purchase_date'].max() - attribution_df['first_purchase_date']
        ).dt.days
        
        attribution_df['is_recent_acquisition'] = (attribution_df['days_since_first_purchase'] <= 30).astype(int)
        attribution_df['acquisition_month'] = attribution_df['first_purchase_date'].dt.month
        attribution_df['acquisition_quarter'] = attribution_df['first_purchase_date'].dt.quarter
        
        # Customer value segments
        attribution_df['clv_quartile'] = pd.qcut(attribution_df['total_customer_value'], 
                                                q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        self.attribution_data = attribution_df
        
        print(f"✅ Attribution data prepared for {len(attribution_df)} customers")
        print(f"📊 Average touchpoints per customer: {attribution_df['touchpoints_count'].mean():.1f}")
        print(f"💰 Average customer value: ${attribution_df['total_customer_value'].mean():.2f}")
        
        return attribution_df
    
    def analyze_channel_performance(self):
        """Analyze performance of different marketing channels"""
        print("📊 Analyzing marketing channel performance...")
        
        if self.attribution_data is None:
            self.prepare_attribution_data()
        
        channel_performance = []
        
        # Get all attribution columns
        attribution_columns = [col for col in self.attribution_data.columns if col.startswith('attribution_')]
        
        for col in attribution_columns:
            channel = col.replace('attribution_', '')
            
            # Customers attributed to this channel
            attributed_customers = self.attribution_data[self.attribution_data[col] > 0]
            
            if len(attributed_customers) == 0:
                continue
            
            # Calculate metrics
            total_customers = len(attributed_customers)
            total_attribution_weight = attributed_customers[col].sum()
            avg_attribution_weight = attributed_customers[col].mean()
            
            # Revenue attribution
            attributed_revenue = (attributed_customers['total_customer_value'] * attributed_customers[col]).sum()
            avg_customer_value = attributed_customers['total_customer_value'].mean()
            
            # Customer quality metrics
            avg_lifetime_days = attributed_customers['customer_lifetime_days'].mean()
            avg_transactions = attributed_customers['total_transactions'].mean()
            premium_customers_pct = (attributed_customers['clv_quartile'] == 'Premium').mean()
            
            # Acquisition efficiency (estimated based on industry benchmarks)
            # Assuming different costs for different channels
            channel_costs = {
                'In-store': 10,
                'Online': 25,
                'Mobile App': 30,
                'Digital_email_open': 5,
                'Digital_website_visit': 15,
                'Digital_product_view': 20,
                'Digital_cart_addition': 35,
                'Digital_search': 12
            }
            
            estimated_cac = channel_costs.get(channel, 20)  # Default CAC
            roi = (attributed_revenue / (total_customers * estimated_cac)) - 1 if total_customers > 0 else 0
            
            # Time to conversion
            avg_days_to_convert = attributed_customers.groupby('customer_id').apply(
                lambda x: (x['first_purchase_date'].iloc[0] - x['first_purchase_date'].iloc[0]).days
            ).mean() if len(attributed_customers) > 0 else 0
            
            channel_performance.append({
                'channel': channel,
                'total_customers': total_customers,
                'total_attribution_weight': total_attribution_weight,
                'avg_attribution_weight': avg_attribution_weight,
                'attributed_revenue': attributed_revenue,
                'avg_customer_value': avg_customer_value,
                'avg_lifetime_days': avg_lifetime_days,
                'avg_transactions_per_customer': avg_transactions,
                'premium_customers_percentage': premium_customers_pct,
                'estimated_cac': estimated_cac,
                'roi': roi,
                'revenue_per_customer': attributed_revenue / total_customers if total_customers > 0 else 0,
                'efficiency_score': (avg_customer_value / estimated_cac) if estimated_cac > 0 else 0
            })
        
        channel_performance_df = pd.DataFrame(channel_performance)
        channel_performance_df = channel_performance_df.sort_values('attributed_revenue', ascending=False)
        
        self.channel_performance = channel_performance_df
        
        print(f"✅ Channel performance analyzed for {len(channel_performance_df)} channels")
        print(f"🏆 Top performing channel by revenue: {channel_performance_df.iloc[0]['channel']}")
        print(f"💰 Best ROI channel: {channel_performance_df.loc[channel_performance_df['roi'].idxmax(), 'channel']}")
        
        return channel_performance_df
    
    def optimize_budget_allocation(self, total_budget=100000):
        """Optimize marketing budget allocation across channels"""
        print("💰 Optimizing marketing budget allocation...")
        
        if self.channel_performance is None:
            self.analyze_channel_performance()
        
        # Current allocation based on attributed revenue
        current_allocation = self.channel_performance.copy()
        current_allocation['current_budget_pct'] = (
            current_allocation['attributed_revenue'] / current_allocation['attributed_revenue'].sum()
        )
        current_allocation['current_budget'] = current_allocation['current_budget_pct'] * total_budget
        
        # Optimization based on efficiency and ROI
        # Weight channels by efficiency score and ROI
        current_allocation['efficiency_weight'] = (
            current_allocation['efficiency_score'] / current_allocation['efficiency_score'].sum()
        ) * 0.6
        
        current_allocation['roi_weight'] = (
            (current_allocation['roi'] + 1) / (current_allocation['roi'] + 1).sum()
        ) * 0.4
        
        current_allocation['optimized_weight'] = (
            current_allocation['efficiency_weight'] + current_allocation['roi_weight']
        )
        
        # Normalize weights
        current_allocation['optimized_budget_pct'] = (
            current_allocation['optimized_weight'] / current_allocation['optimized_weight'].sum()
        )
        current_allocation['optimized_budget'] = current_allocation['optimized_budget_pct'] * total_budget
        
        # Calculate budget changes
        current_allocation['budget_change'] = (
            current_allocation['optimized_budget'] - current_allocation['current_budget']
        )
        current_allocation['budget_change_pct'] = (
            current_allocation['budget_change'] / current_allocation['current_budget']
        ).fillna(0)
        
        # Recommendations
        increase_budget = current_allocation[current_allocation['budget_change'] > 1000].sort_values('budget_change', ascending=False)
        decrease_budget = current_allocation[current_allocation['budget_change'] < -1000].sort_values('budget_change')
        
        optimization_results = {
            'total_budget': total_budget,
            'channel_allocations': current_allocation.to_dict('records'),
            'recommendations': {
                'increase_budget': increase_budget[['channel', 'budget_change', 'roi', 'efficiency_score']].to_dict('records'),
                'decrease_budget': decrease_budget[['channel', 'budget_change', 'roi', 'efficiency_score']].to_dict('records')
            },
            'expected_improvement': {
                'roi_improvement': (current_allocation['roi'] * current_allocation['optimized_budget_pct']).sum() - 
                                 (current_allocation['roi'] * current_allocation['current_budget_pct']).sum(),
                'efficiency_improvement': (current_allocation['efficiency_score'] * current_allocation['optimized_budget_pct']).sum() - 
                                        (current_allocation['efficiency_score'] * current_allocation['current_budget_pct']).sum()
            }
        }
        
        print(f"✅ Budget optimization completed")
        print(f"📈 Expected ROI improvement: {optimization_results['expected_improvement']['roi_improvement']:.2%}")
        print(f"⚡ Expected efficiency improvement: {optimization_results['expected_improvement']['efficiency_improvement']:.2%}")
        
        return optimization_results
    
    def analyze_customer_acquisition_funnel(self):
        """Analyze customer acquisition funnel and conversion rates"""
        print("🔄 Analyzing customer acquisition funnel...")
        
        # Digital events funnel analysis
        funnel_stages = ['website_visit', 'product_view', 'cart_addition', 'purchase_intent', 'email_open']
        
        funnel_data = []
        
        for stage in funnel_stages:
            stage_events = self.digital_events[self.digital_events['event_type'] == stage]
            unique_customers = stage_events['customer_id'].nunique()
            total_events = len(stage_events)
            
            # Conversion to purchase
            customers_with_stage = stage_events['customer_id'].unique()
            customers_who_purchased = self.transactions[
                self.transactions['customer_id'].isin(customers_with_stage)
            ]['customer_id'].nunique()
            
            conversion_rate = customers_who_purchased / unique_customers if unique_customers > 0 else 0
            
            # Average time to conversion
            stage_customers = stage_events.groupby('customer_id')['event_date'].min().reset_index()
            stage_customers.columns = ['customer_id', 'stage_date']
            
            first_purchases = self.transactions.groupby('customer_id')['transaction_date'].min().reset_index()
            first_purchases.columns = ['customer_id', 'first_purchase_date']
            
            conversion_times = stage_customers.merge(first_purchases, on='customer_id', how='inner')
            conversion_times['days_to_convert'] = (
                conversion_times['first_purchase_date'] - conversion_times['stage_date']
            ).dt.days
            
            avg_days_to_convert = conversion_times['days_to_convert'].mean() if len(conversion_times) > 0 else 0
            
            funnel_data.append({
                'stage': stage,
                'unique_customers': unique_customers,
                'total_events': total_events,
                'customers_who_purchased': customers_who_purchased,
                'conversion_rate': conversion_rate,
                'avg_days_to_convert': avg_days_to_convert,
                'events_per_customer': total_events / unique_customers if unique_customers > 0 else 0
            })
        
        funnel_df = pd.DataFrame(funnel_data)
        funnel_df = funnel_df.sort_values('conversion_rate', ascending=False)
        
        print(f"✅ Funnel analysis completed for {len(funnel_stages)} stages")
        print(f"🏆 Best converting stage: {funnel_df.iloc[0]['stage']} ({funnel_df.iloc[0]['conversion_rate']:.1%})")
        
        return funnel_df
    
    def create_marketing_dashboard_data(self):
        """Create comprehensive data for marketing attribution dashboard"""
        channel_performance = self.analyze_channel_performance()
        budget_optimization = self.optimize_budget_allocation()
        funnel_analysis = self.analyze_customer_acquisition_funnel()
        
        dashboard_data = {
            'attribution_summary': {
                'total_customers_analyzed': len(self.attribution_data),
                'total_attributed_revenue': channel_performance['attributed_revenue'].sum(),
                'avg_customer_value': self.attribution_data['total_customer_value'].mean(),
                'avg_touchpoints_per_customer': self.attribution_data['touchpoints_count'].mean(),
                'top_performing_channel': channel_performance.iloc[0]['channel'],
                'best_roi_channel': channel_performance.loc[channel_performance['roi'].idxmax(), 'channel']
            },
            'channel_performance': channel_performance.to_dict('records'),
            'budget_optimization': budget_optimization,
            'funnel_analysis': funnel_analysis.to_dict('records'),
            'key_insights': [
                f"Top channel by revenue: {channel_performance.iloc[0]['channel']} (${channel_performance.iloc[0]['attributed_revenue']:,.0f})",
                f"Best ROI channel: {channel_performance.loc[channel_performance['roi'].idxmax(), 'channel']} ({channel_performance['roi'].max():.1%} ROI)",
                f"Average customer acquisition cost: ${channel_performance['estimated_cac'].mean():.0f}",
                f"Premium customer rate: {(self.attribution_data['clv_quartile'] == 'Premium').mean():.1%}",
                f"Multi-touch customers: {(self.attribution_data['touchpoints_count'] > 1).mean():.1%}"
            ]
        }
        
        return dashboard_data

def main():
    """Main execution function"""
    print("🚀 MARKETING ATTRIBUTION & ROI ANALYSIS ENGINE")
    print("=" * 60)
    
    # Initialize marketing attribution engine
    attribution_engine = MarketingAttributionEngine(
        transactions_path='data/transactions.csv',
        customers_path='data/customers.csv',
        digital_events_path='data/digital_events.csv'
    )
    
    # Prepare attribution data
    attribution_engine.prepare_attribution_data()
    
    # Analyze channel performance
    channel_performance = attribution_engine.analyze_channel_performance()
    
    # Optimize budget allocation
    budget_optimization = attribution_engine.optimize_budget_allocation()
    
    # Analyze acquisition funnel
    funnel_analysis = attribution_engine.analyze_customer_acquisition_funnel()
    
    # Save results
    attribution_engine.attribution_data.to_csv('results/marketing_attribution_data.csv', index=False)
    channel_performance.to_csv('results/channel_performance_analysis.csv', index=False)
    funnel_analysis.to_csv('results/acquisition_funnel_analysis.csv', index=False)
    
    # Create dashboard data
    dashboard_data = attribution_engine.create_marketing_dashboard_data()
    
    import json
    with open('results/marketing_attribution_dashboard.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    with open('results/budget_optimization_recommendations.json', 'w') as f:
        json.dump(budget_optimization, f, indent=2, default=str)
    
    print("\n✅ Marketing attribution analysis completed!")
    print("📁 Results saved to:")
    print("   - results/marketing_attribution_data.csv")
    print("   - results/channel_performance_analysis.csv")
    print("   - results/acquisition_funnel_analysis.csv")
    print("   - results/marketing_attribution_dashboard.json")
    print("   - results/budget_optimization_recommendations.json")

if __name__ == "__main__":
    main()