"""
Customer Segmentation using RFM Analysis and Advanced Clustering
Implements state-of-the-art segmentation techniques for Coop Norge business case
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Advanced Customer Segmentation for Retail Cooperative Business
    
    Features:
    - RFM Analysis (Recency, Frequency, Monetary)
    - Multiple clustering algorithms
    - Automated cluster validation
    - Business-friendly segment naming
    - Actionable insights generation
    """
    
    def __init__(self, transaction_data_path, customer_data_path):
        """Initialize with transaction and customer data paths"""
        self.transactions = pd.read_csv(transaction_data_path)
        self.customers = pd.read_csv(customer_data_path)
        self.rfm_data = None
        self.segments = None
        self.model = None
        
    def prepare_data(self):
        """Prepare and clean data for segmentation"""
        print("🔄 Preparing data for segmentation...")
        
        # Convert date columns
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        
        # Filter out any negative amounts or quantities
        self.transactions = self.transactions[
            (self.transactions['total_amount'] > 0) & 
            (self.transactions['quantity'] > 0)
        ]
        
        # Calculate reference date (most recent transaction + 1 day)
        self.reference_date = self.transactions['transaction_date'].max() + timedelta(days=1)
        
        print(f"✅ Data prepared: {len(self.transactions)} transactions, {len(self.customers)} customers")
        
    def calculate_rfm(self):
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        print("📊 Calculating RFM metrics...")
        
        # Calculate RFM metrics
        rfm = self.transactions.groupby('customer_id').agg({
            'transaction_date': [
                ('recency', lambda x: (self.reference_date - x.max()).days),
                ('frequency', 'count')
            ],
            'total_amount': [
                ('monetary', 'sum'),
                ('avg_order_value', 'mean')
            ]
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary', 'avg_order_value']
        rfm = rfm.reset_index()
        
        # Add customer demographics
        rfm = rfm.merge(self.customers[['customer_id', 'loyalty_tier', 'age', 'household_size']], 
                       on='customer_id', how='left')
        
        # Calculate additional metrics
        rfm['days_since_first_purchase'] = self.transactions.groupby('customer_id')['transaction_date'].agg(
            lambda x: (self.reference_date - x.min()).days
        ).values
        
        rfm['purchase_frequency_per_month'] = rfm['frequency'] / (rfm['days_since_first_purchase'] / 30)
        
        # Proper CLV calculation with forward-looking components
        # CLV = (Average Order Value × Purchase Frequency × Gross Margin × Customer Lifespan) - Acquisition Cost
        
        # Calculate average time between purchases (in days)
        avg_days_between_purchases = rfm['days_since_first_purchase'] / (rfm['frequency'] + 1)
        
        # Estimate annual purchase frequency
        annual_frequency = 365 / avg_days_between_purchases.replace([np.inf, 0], 365)  # Handle edge cases
        
        # Calculate average order value
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        rfm['avg_order_value'] = rfm['avg_order_value'].fillna(rfm['monetary'])  # For single purchases
        
        # Estimate customer lifespan (simplified model based on recency and frequency)
        # More frequent, recent customers likely to have longer lifespan
        recency_factor = 1 / (1 + rfm['recency'] / 365)  # Decay function
        frequency_factor = np.log1p(rfm['frequency']) / np.log1p(rfm['frequency'].max())
        estimated_lifespan_years = 2 * recency_factor * frequency_factor + 0.5  # 0.5 to 2.5 years
        
        # Gross margin assumption (typical retail 20-30%)
        gross_margin = 0.25
        
        # Acquisition cost assumption (typical retail 50-100 NOK)
        acquisition_cost = 75
        
        # Calculate forward-looking CLV
        rfm['customer_lifetime_value'] = (
            rfm['avg_order_value'] * 
            annual_frequency * 
            gross_margin * 
            estimated_lifespan_years
        ) - acquisition_cost
        
        # Ensure CLV is positive (minimum 0)
        rfm['customer_lifetime_value'] = np.maximum(rfm['customer_lifetime_value'], 0)
        
        # Handle any missing values
        rfm = rfm.fillna(0)
        
        self.rfm_data = rfm
        print(f"✅ RFM metrics calculated for {len(rfm)} customers")
        
        return rfm
    
    def create_rfm_scores(self):
        """Create RFM scores using quantile-based scoring"""
        print("🎯 Creating RFM scores...")
        
        if self.rfm_data is None:
            self.calculate_rfm()
        
        # Create RFM scores (1-5 scale) with robust binning
        # Note: For Recency, lower values are better (more recent)
        try:
            self.rfm_data['recency_score'] = pd.qcut(self.rfm_data['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data['recency_score'] = pd.cut(self.rfm_data['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        
        try:
            self.rfm_data['frequency_score'] = pd.qcut(self.rfm_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data['frequency_score'] = pd.cut(self.rfm_data['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        try:
            self.rfm_data['monetary_score'] = pd.qcut(self.rfm_data['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data['monetary_score'] = pd.cut(self.rfm_data['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        self.rfm_data['recency_score'] = self.rfm_data['recency_score'].astype(int)
        self.rfm_data['frequency_score'] = self.rfm_data['frequency_score'].astype(int)
        self.rfm_data['monetary_score'] = self.rfm_data['monetary_score'].astype(int)
        
        # Create combined RFM score
        self.rfm_data['rfm_score'] = (
            self.rfm_data['recency_score'].astype(str) + 
            self.rfm_data['frequency_score'].astype(str) + 
            self.rfm_data['monetary_score'].astype(str)
        )
        
        # Create segment based on RFM score
        self.rfm_data['rfm_segment'] = self.rfm_data['rfm_score'].apply(self._categorize_rfm)
        
        print("✅ RFM scores created")
        return self.rfm_data
    
    def _categorize_rfm(self, rfm_score):
        """Categorize customers based on RFM score"""
        if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif rfm_score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return 'Potential Loyalists'
        elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif rfm_score in ['155', '254', '144', '214', '215', '115', '114']:
            return 'Cannot Lose Them'
        elif rfm_score in ['331', '321', '231', '241', '251']:
            return 'Hibernating'
        else:
            return 'Others'
    
    def advanced_clustering(self, n_clusters=None):
        """Perform advanced clustering using multiple algorithms"""
        print("🤖 Performing advanced clustering...")
        
        if self.rfm_data is None:
            self.create_rfm_scores()
        
        # Prepare features for clustering
        features = ['recency', 'frequency', 'monetary', 'avg_order_value', 'purchase_frequency_per_month']
        X = self.rfm_data[features].copy()
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Apply Gaussian Mixture Model for comparison
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        self.rfm_data['cluster_gmm'] = gmm.fit_predict(X_scaled)
        
        # Evaluate clustering quality
        silhouette_kmeans = silhouette_score(X_scaled, self.rfm_data['cluster'])
        silhouette_gmm = silhouette_score(X_scaled, self.rfm_data['cluster_gmm'])
        
        # Choose best clustering method
        if silhouette_kmeans >= silhouette_gmm:
            self.rfm_data['final_cluster'] = self.rfm_data['cluster']
            self.model = kmeans
            print(f"✅ K-Means selected (Silhouette Score: {silhouette_kmeans:.3f})")
        else:
            self.rfm_data['final_cluster'] = self.rfm_data['cluster_gmm']
            self.model = gmm
            print(f"✅ GMM selected (Silhouette Score: {silhouette_gmm:.3f})")
        
        # Create business-friendly segment names
        self._create_segment_names()
        
        return self.rfm_data
    
    def _find_optimal_clusters(self, X_scaled, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(X_scaled) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"🎯 Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _create_segment_names(self):
        """Create business-friendly names for clusters based on characteristics"""
        cluster_summary = self.rfm_data.groupby('final_cluster').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean',
            'customer_lifetime_value': 'mean'
        }).round(2)
        
        # Name segments based on characteristics
        segment_names = {}
        for cluster in cluster_summary.index:
            row = cluster_summary.loc[cluster]
            
            if row['monetary'] > cluster_summary['monetary'].quantile(0.8):
                if row['frequency'] > cluster_summary['frequency'].quantile(0.7):
                    segment_names[cluster] = 'VIP Champions'
                else:
                    segment_names[cluster] = 'Big Spenders'
            elif row['frequency'] > cluster_summary['frequency'].quantile(0.8):
                segment_names[cluster] = 'Frequent Buyers'
            elif row['recency'] < cluster_summary['recency'].quantile(0.3):
                segment_names[cluster] = 'Recent Customers'
            elif row['recency'] > cluster_summary['recency'].quantile(0.7):
                segment_names[cluster] = 'At Risk'
            else:
                segment_names[cluster] = f'Regular Customers'
        
        self.rfm_data['segment_name'] = self.rfm_data['final_cluster'].map(segment_names)
        
    def generate_insights(self):
        """Generate actionable business insights from segmentation"""
        print("💡 Generating business insights...")
        
        if self.rfm_data is None:
            self.advanced_clustering()
        
        insights = {}
        
        # Overall statistics
        insights['total_customers'] = len(self.rfm_data)
        insights['total_revenue'] = self.rfm_data['monetary'].sum()
        insights['avg_clv'] = self.rfm_data['customer_lifetime_value'].mean()
        
        # Segment analysis
        segment_analysis = self.rfm_data.groupby('segment_name').agg({
            'customer_id': 'count',
            'monetary': ['sum', 'mean'],
            'frequency': 'mean',
            'recency': 'mean',
            'customer_lifetime_value': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['customers', 'total_revenue', 'avg_revenue', 'avg_frequency', 'avg_recency', 'avg_clv']
        segment_analysis['revenue_percentage'] = (segment_analysis['total_revenue'] / segment_analysis['total_revenue'].sum() * 100).round(1)
        segment_analysis['customer_percentage'] = (segment_analysis['customers'] / segment_analysis['customers'].sum() * 100).round(1)
        
        insights['segment_analysis'] = segment_analysis
        
        # Top segments by revenue
        insights['top_revenue_segments'] = segment_analysis.nlargest(3, 'total_revenue').index.tolist()
        
        # At-risk customers
        at_risk = self.rfm_data[self.rfm_data['recency'] > 90]  # Haven't purchased in 90+ days
        insights['at_risk_customers'] = len(at_risk)
        insights['at_risk_revenue'] = at_risk['monetary'].sum()
        
        # Loyalty tier analysis
        loyalty_analysis = self.rfm_data.groupby('loyalty_tier').agg({
            'monetary': ['mean', 'sum'],
            'frequency': 'mean'
        }).round(2)
        
        insights['loyalty_analysis'] = loyalty_analysis
        
        self.insights = insights
        return insights
    
    def plot_segmentation_analysis(self, save_path=None):
        """Create comprehensive segmentation visualizations"""
        if self.rfm_data is None:
            self.advanced_clustering()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Coop Norge - Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. RFM Distribution
        axes[0,0].hist(self.rfm_data['recency'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Recency Distribution')
        axes[0,0].set_xlabel('Days Since Last Purchase')
        
        axes[0,1].hist(self.rfm_data['frequency'], bins=30, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Frequency Distribution')  
        axes[0,1].set_xlabel('Number of Purchases')
        
        axes[0,2].hist(self.rfm_data['monetary'], bins=30, alpha=0.7, color='salmon')
        axes[0,2].set_title('Monetary Distribution')
        axes[0,2].set_xlabel('Total Spent (NOK)')
        
        # 2. Segment Analysis
        segment_counts = self.rfm_data['segment_name'].value_counts()
        axes[1,0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Customer Segments Distribution')
        
        # 3. Segment Revenue
        segment_revenue = self.rfm_data.groupby('segment_name')['monetary'].sum().sort_values(ascending=True)
        axes[1,1].barh(segment_revenue.index, segment_revenue.values)
        axes[1,1].set_title('Revenue by Segment (NOK)')
        axes[1,1].set_xlabel('Total Revenue')
        
        # 4. RFM Scatter
        scatter = axes[1,2].scatter(self.rfm_data['frequency'], self.rfm_data['monetary'], 
                                  c=self.rfm_data['final_cluster'], cmap='viridis', alpha=0.6)
        axes[1,2].set_title('Frequency vs Monetary by Cluster')
        axes[1,2].set_xlabel('Frequency')
        axes[1,2].set_ylabel('Monetary (NOK)')
        plt.colorbar(scatter, ax=axes[1,2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis plots saved to {save_path}")
        
        plt.show()
    
    def get_marketing_recommendations(self):
        """Generate specific marketing recommendations for each segment"""
        if not hasattr(self, 'insights'):
            self.generate_insights()
        
        recommendations = {}
        
        for segment in self.rfm_data['segment_name'].unique():
            segment_data = self.rfm_data[self.rfm_data['segment_name'] == segment]
            avg_recency = segment_data['recency'].mean()
            avg_frequency = segment_data['frequency'].mean()
            avg_monetary = segment_data['monetary'].mean()
            
            if 'VIP' in segment or 'Champions' in segment:
                recommendations[segment] = {
                    'strategy': 'Retain and Reward',
                    'tactics': [
                        'Exclusive member events and early access',
                        'Premium customer service',
                        'Personalized high-value product recommendations',
                        'Loyalty program upgrades'
                    ],
                    'channel': 'Personal touch + digital',
                    'frequency': 'Monthly'
                }
            elif 'At Risk' in segment:
                recommendations[segment] = {
                    'strategy': 'Win-back Campaign',
                    'tactics': [
                        'Discount coupons for next purchase',
                        'Email series highlighting new products',
                        'Survey to understand issues',
                        'Limited-time offers'
                    ],
                    'channel': 'Email + SMS',
                    'frequency': 'Bi-weekly'
                }
            elif 'Recent' in segment or 'New' in segment:
                recommendations[segment] = {
                    'strategy': 'Onboarding and Education',
                    'tactics': [
                        'Welcome series explaining Coop benefits',
                        'Tutorial on loyalty program',
                        'Cross-selling complementary products',
                        'First purchase incentives'
                    ],
                    'channel': 'Mobile app + email',
                    'frequency': 'Weekly for first month'
                }
            else:
                recommendations[segment] = {
                    'strategy': 'Engagement and Growth',
                    'tactics': [
                        'Product recommendations based on purchase history',
                        'Seasonal campaign participation',
                        'Category-specific promotions',
                        'Loyalty point boosters'
                    ],
                    'channel': 'App notifications + email',
                    'frequency': 'Bi-weekly'
                }
        
        return recommendations
    
    def export_segments(self, filename='customer_segments.csv'):
        """Export segmentation results for business use"""
        if self.rfm_data is None:
            self.advanced_clustering()
        
        # Dynamically select available columns
        required_cols = ['customer_id', 'recency', 'frequency', 'monetary']
        optional_cols = ['segment_name', 'final_cluster', 'customer_lifetime_value', 
                        'rfm_score', 'loyalty_tier', 'age', 'household_size']
        
        export_cols = required_cols.copy()
        for col in optional_cols:
            if col in self.rfm_data.columns:
                export_cols.append(col)
        
        export_data = self.rfm_data[export_cols].copy()
        
        export_data.to_csv(filename, index=False)
        print(f"📤 Segmentation results exported to {filename}")
        print(f"   Columns exported: {', '.join(export_cols)}")
        return export_data

if __name__ == "__main__":
    # Example usage
    print("🚀 Coop Norge - Customer Segmentation Analysis")
    print("=" * 50)
    
    # Initialize segmentation
    segmentation = CustomerSegmentation('../data/transactions.csv', '../data/customers.csv')
    
    # Run complete analysis
    segmentation.prepare_data()
    segmentation.calculate_rfm()
    segmentation.create_rfm_scores()
    segmentation.advanced_clustering()
    
    # Generate insights
    insights = segmentation.generate_insights()
    print(f"\n📈 KEY INSIGHTS:")
    print(f"• Total Customers: {insights['total_customers']:,}")
    print(f"• Total Revenue: {insights['total_revenue']:,.2f} NOK")
    print(f"• Average CLV: {insights['avg_clv']:,.2f} NOK")
    print(f"• At-Risk Customers: {insights['at_risk_customers']:,}")
    
    # Get marketing recommendations
    recommendations = segmentation.get_marketing_recommendations()
    print(f"\n🎯 MARKETING RECOMMENDATIONS GENERATED FOR {len(recommendations)} SEGMENTS")
    
    # Export results
    segmentation.export_segments('../results/customer_segments.csv')
    
    # Create visualizations
    segmentation.plot_segmentation_analysis('../results/segmentation_analysis.png') 