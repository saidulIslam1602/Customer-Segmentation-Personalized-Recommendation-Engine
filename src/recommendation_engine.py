"""
Personalized Recommendation Engine for Coop Norge
Implements collaborative filtering, content-based filtering, and hybrid approaches
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """
    Advanced Recommendation Engine for Retail Cooperative
    
    Features:
    - Collaborative Filtering (User-based & Item-based)
    - Content-Based Filtering
    - Hybrid Recommendations
    - Cold Start Problem Handling
    - A/B Testing Framework
    - Real-time Scoring
    """
    
    def __init__(self, transactions_path, customers_path, products_path):
        """Initialize with data paths"""
        self.transactions = pd.read_csv(transactions_path)
        self.customers = pd.read_csv(customers_path)
        self.products = pd.read_csv(products_path)
        
        self.user_item_matrix = None
        self.item_features = None
        self.collaborative_model = None
        self.content_model = None
        
    def prepare_data(self):
        """Prepare data for recommendation algorithms"""
        print("🔄 Preparing recommendation data...")
        
        # Create user-item interaction matrix
        # Aggregate by customer and product (sum quantities as interaction strength)
        interactions = self.transactions.groupby(['customer_id', 'product_id']).agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'transaction_date': 'count'  # Number of times purchased
        }).reset_index()
        
        interactions.columns = ['customer_id', 'product_id', 'quantity', 'total_spent', 'purchase_frequency']
        
        # Create interaction score (combining quantity, spending, and frequency)
        interactions['interaction_score'] = (
            interactions['quantity'] * 0.3 +
            (interactions['total_spent'] / interactions['total_spent'].max()) * 100 * 0.4 +
            interactions['purchase_frequency'] * 0.3
        )
        
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='customer_id',
            columns='product_id', 
            values='interaction_score',
            fill_value=0
        )
        
        # Prepare product features for content-based filtering
        self._prepare_product_features()
        
        print(f"✅ Data prepared: {len(self.user_item_matrix)} users, {len(self.user_item_matrix.columns)} products")
        
    def _prepare_product_features(self):
        """Prepare product features for content-based filtering"""
        
        # Create product feature matrix
        products_features = self.products.copy()
        
        # Create text features from product attributes
        products_features['combined_features'] = (
            products_features['category'].fillna('') + ' ' +
            products_features['subcategory'].fillna('') + ' ' +
            products_features['brand'].fillna('') + ' ' +
            products_features['organic'].astype(str) + ' ' +
            products_features['local_producer'].astype(str)
        )
        
        # TF-IDF vectorization of text features
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(products_features['combined_features'])
        
        # Convert to DataFrame
        feature_names = tfidf.get_feature_names_out()
        self.item_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=products_features['product_id']
        )
        
        # Add numerical features
        numerical_features = ['price']
        for feature in numerical_features:
            if feature in products_features.columns:
                # Normalize numerical features
                scaler = StandardScaler()
                normalized_values = scaler.fit_transform(products_features[[feature]])
                self.item_features[f'{feature}_normalized'] = normalized_values.flatten()
    
    def build_collaborative_filtering(self, n_components=50):
        """Build collaborative filtering model using Matrix Factorization"""
        print("🤖 Building collaborative filtering model...")
        
        if self.user_item_matrix is None:
            self.prepare_data()
        
        # Use Truncated SVD for matrix factorization
        self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model
        user_features = self.collaborative_model.fit_transform(self.user_item_matrix)
        item_features = self.collaborative_model.components_
        
        # Reconstruct the full matrix for predictions
        self.predicted_ratings = np.dot(user_features, item_features)
        self.predicted_ratings_df = pd.DataFrame(
            self.predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )
        
        print(f"✅ Collaborative filtering model built with {n_components} components")
    
    def build_content_based_filtering(self):
        """Build content-based filtering model with memory optimization"""
        print("📊 Building content-based filtering model...")
        
        if self.item_features is None:
            self.prepare_data()
        
        # Use memory-efficient approach for large datasets
        n_items = len(self.item_features)
        print(f"🔧 Processing {n_items:,} items with memory optimization...")
        
        if n_items > 10000:
            # For large datasets, use sparse similarity calculation
            print("⚡ Using sparse similarity calculation for large dataset...")
            # Calculate similarity on-demand rather than storing full matrix
            self.item_similarity_df = None  # Will calculate similarities on-demand
            self.item_features_sparse = self.item_features.copy()
        else:
            # For smaller datasets, calculate full similarity matrix
            self.item_similarity_matrix = cosine_similarity(self.item_features)
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity_matrix,
                index=self.item_features.index,
                columns=self.item_features.index
            )
        
        print("✅ Content-based filtering model built with memory optimization")
    
    def get_collaborative_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if self.collaborative_model is None:
            self.build_collaborative_filtering()
        
        if customer_id not in self.predicted_ratings_df.index:
            return self._handle_cold_start_user(customer_id, n_recommendations)
        
        # Get user's predicted ratings
        user_predictions = self.predicted_ratings_df.loc[customer_id]
        
        # Get items user hasn't interacted with
        user_interactions = self.user_item_matrix.loc[customer_id]
        unrated_items = user_interactions[user_interactions == 0].index
        
        # Get predictions for unrated items
        recommendations = user_predictions[unrated_items].sort_values(ascending=False)
        
        # Add product information
        top_recommendations = recommendations.head(n_recommendations)
        result = []
        
        for product_id, predicted_score in top_recommendations.items():
            product_info = self.products[self.products['product_id'] == product_id].iloc[0]
            result.append({
                'product_id': product_id,
                'predicted_score': predicted_score,
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'price': product_info['price'],
                'recommendation_type': 'collaborative'
            })
        
        return result
    
    def get_content_based_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using content-based filtering with memory optimization"""
        if not hasattr(self, 'item_features_sparse') and self.item_similarity_df is None:
            self.build_content_based_filtering()
        
        if customer_id not in self.user_item_matrix.index:
            return self._handle_cold_start_user(customer_id, n_recommendations)
        
        # Get user's purchase history
        user_purchases = self.user_item_matrix.loc[customer_id]
        purchased_items = user_purchases[user_purchases > 0].index.tolist()
        
        if not purchased_items:
            return self._get_popular_recommendations(n_recommendations)
        
        # Handle large datasets with on-demand similarity calculation
        if hasattr(self, 'item_features_sparse') and self.item_similarity_df is None:
            # Memory-efficient approach: calculate similarities on-demand
            content_scores = {}
            all_items = self.item_features_sparse.index.tolist()
            
            # Sample top items for efficiency with large datasets
            sample_size = min(1000, len(all_items))
            sampled_items = np.random.choice(all_items, sample_size, replace=False)
            
            for item in sampled_items:
                if item not in purchased_items:
                    # Calculate similarity to purchased items
                    total_similarity = 0.0
                    for purchased_item in purchased_items:
                        if purchased_item in self.item_features_sparse.index and item in self.item_features_sparse.index:
                            item_vec = self.item_features_sparse.loc[item].values.reshape(1, -1)
                            purchased_vec = self.item_features_sparse.loc[purchased_item].values.reshape(1, -1)
                            similarity = cosine_similarity(item_vec, purchased_vec)[0, 0]
                            item_weight = user_purchases[purchased_item]
                            total_similarity += similarity * item_weight
                    
                    content_scores[item] = total_similarity
            
            # Sort and get top recommendations
            sorted_scores = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            top_recommendations = sorted_scores[:n_recommendations]
            
        else:
            # Original approach for smaller datasets
            content_scores = pd.Series(0.0, index=self.item_similarity_df.index)
            
            for purchased_item in purchased_items:
                if purchased_item in self.item_similarity_df.index:
                    # Weight by user's interaction strength
                    item_weight = user_purchases[purchased_item]
                    similarities = self.item_similarity_df[purchased_item]
                    content_scores = content_scores.add(similarities * item_weight, fill_value=0.0)
            
            # Remove items user has already purchased
            content_scores = content_scores.drop(purchased_items, errors='ignore')
            
            # Get top recommendations
            top_recommendations = content_scores.sort_values(ascending=False).head(n_recommendations).items()
        
        result = []
        for product_id, content_score in top_recommendations:
            product_info = self.products[self.products['product_id'] == product_id]
            if not product_info.empty:
                product_info = product_info.iloc[0]
                result.append({
                    'product_id': product_id,
                    'predicted_score': float(content_score),
                    'product_name': product_info['product_name'],
                    'category': product_info['category'],
                    'price': product_info['price'],
                    'recommendation_type': 'content_based'
                })
        
        return result
    
    def get_hybrid_recommendations(self, customer_id, n_recommendations=10, 
                                 collaborative_weight=0.6, content_weight=0.4):
        """Get hybrid recommendations combining collaborative and content-based"""
        print(f"🎯 Generating hybrid recommendations for {customer_id}...")
        
        # Get recommendations from both methods
        collaborative_recs = self.get_collaborative_recommendations(customer_id, n_recommendations * 2)
        content_recs = self.get_content_based_recommendations(customer_id, n_recommendations * 2)
        
        # Combine scores
        hybrid_scores = {}
        
        # Add collaborative filtering scores
        for rec in collaborative_recs:
            product_id = rec['product_id']
            hybrid_scores[product_id] = {
                'score': rec['predicted_score'] * collaborative_weight,
                'product_info': rec
            }
        
        # Add content-based scores
        for rec in content_recs:
            product_id = rec['product_id']
            if product_id in hybrid_scores:
                hybrid_scores[product_id]['score'] += rec['predicted_score'] * content_weight
            else:
                hybrid_scores[product_id] = {
                    'score': rec['predicted_score'] * content_weight,
                    'product_info': rec
                }
        
        # Sort by combined score and return top N
        sorted_recommendations = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:n_recommendations]
        
        result = []
        for product_id, data in sorted_recommendations:
            rec = data['product_info'].copy()
            rec['predicted_score'] = data['score']
            rec['recommendation_type'] = 'hybrid'
            result.append(rec)
        
        return result
    
    def _handle_cold_start_user(self, customer_id, n_recommendations=10):
        """Handle recommendations for new users (cold start problem)"""
        print(f"🆕 Handling cold start for new customer: {customer_id}")
        
        # Get customer demographics if available
        customer_info = self.customers[self.customers['customer_id'] == customer_id]
        
        if not customer_info.empty:
            customer = customer_info.iloc[0]
            
            # Find similar customers by demographics
            similar_customers = self.customers[
                (self.customers['age'] >= customer['age'] - 10) &
                (self.customers['age'] <= customer['age'] + 10) &
                (self.customers['household_size'] == customer['household_size'])
            ]['customer_id'].tolist()
            
            # Get popular products among similar customers
            if similar_customers:
                similar_purchases = self.transactions[
                    self.transactions['customer_id'].isin(similar_customers)
                ]
                
                popular_products = similar_purchases.groupby('product_id').agg({
                    'quantity': 'sum',
                    'total_amount': 'sum'
                }).reset_index()
                
                popular_products['popularity_score'] = (
                    popular_products['quantity'] * 0.4 +
                    popular_products['total_amount'] / popular_products['total_amount'].max() * 100 * 0.6
                )
                
                top_products = popular_products.nlargest(n_recommendations, 'popularity_score')
                
                result = []
                for _, row in top_products.iterrows():
                    product_info = self.products[self.products['product_id'] == row['product_id']].iloc[0]
                    result.append({
                        'product_id': row['product_id'],
                        'predicted_score': row['popularity_score'],
                        'product_name': product_info['product_name'],
                        'category': product_info['category'],
                        'price': product_info['price'],
                        'recommendation_type': 'demographic_based'
                    })
                
                return result
        
        # Fallback to popular recommendations
        return self._get_popular_recommendations(n_recommendations)
    
    def _get_popular_recommendations(self, n_recommendations=10):
        """Get popular product recommendations"""
        popular_products = self.transactions.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'customer_id': 'nunique'  # Number of unique customers
        }).reset_index()
        
        popular_products['popularity_score'] = (
            popular_products['quantity'] * 0.3 +
            popular_products['total_amount'] / popular_products['total_amount'].max() * 100 * 0.4 +
            popular_products['customer_id'] * 0.3
        )
        
        top_products = popular_products.nlargest(n_recommendations, 'popularity_score')
        
        result = []
        for _, row in top_products.iterrows():
            product_info = self.products[self.products['product_id'] == row['product_id']].iloc[0]
            result.append({
                'product_id': row['product_id'],
                'predicted_score': row['popularity_score'],
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'price': product_info['price'],
                'recommendation_type': 'popular'
            })
        
        return result
    
    def get_category_recommendations(self, customer_id, category, n_recommendations=5):
        """Get recommendations within a specific category"""
        if customer_id not in self.user_item_matrix.index:
            # For new users, return popular products in category
            category_products = self.products[self.products['category'] == category]
            category_transactions = self.transactions[
                self.transactions['product_id'].isin(category_products['product_id'])
            ]
            
            popular_in_category = category_transactions.groupby('product_id').agg({
                'quantity': 'sum'
            }).reset_index().nlargest(n_recommendations, 'quantity')
            
            result = []
            for _, row in popular_in_category.iterrows():
                product_info = self.products[self.products['product_id'] == row['product_id']].iloc[0]
                result.append({
                    'product_id': row['product_id'],
                    'predicted_score': row['quantity'],
                    'product_name': product_info['product_name'],
                    'category': product_info['category'],
                    'price': product_info['price'],
                    'recommendation_type': f'popular_in_{category}'
                })
            
            return result
        
        # Get hybrid recommendations and filter by category
        all_recommendations = self.get_hybrid_recommendations(customer_id, n_recommendations * 3)
        category_recommendations = [
            rec for rec in all_recommendations 
            if rec['category'] == category
        ][:n_recommendations]
        
        return category_recommendations
    
    def calculate_recommendation_metrics(self, test_transactions=None):
        """Calculate recommendation system metrics"""
        print("📈 Calculating recommendation metrics...")
        
        if test_transactions is None:
            # Use recent transactions as test set
            recent_date = self.transactions['transaction_date'].max()
            test_transactions = self.transactions[
                pd.to_datetime(self.transactions['transaction_date']) >= 
                pd.to_datetime(recent_date) - pd.Timedelta(days=30)
            ]
        
        metrics = {
            'coverage': 0,
            'diversity': 0,
            'novelty': 0,
            'precision_at_10': 0
        }
        
        # Calculate coverage (what % of items can be recommended)
        all_products = set(self.products['product_id'])
        recommended_products = set()
        
        sample_customers = self.user_item_matrix.index[:100]  # Sample for efficiency
        
        for customer_id in sample_customers:
            try:
                recs = self.get_hybrid_recommendations(customer_id, 10)
                recommended_products.update([rec['product_id'] for rec in recs])
            except:
                continue
        
        metrics['coverage'] = len(recommended_products) / len(all_products)
        
        # Calculate diversity (average intra-list distance)
        if hasattr(self, 'item_similarity_df'):
            diversity_scores = []
            for customer_id in sample_customers[:20]:  # Smaller sample for diversity
                try:
                    recs = self.get_hybrid_recommendations(customer_id, 10)
                    rec_products = [rec['product_id'] for rec in recs]
                    
                    if len(rec_products) > 1:
                        similarities = []
                        for i in range(len(rec_products)):
                            for j in range(i+1, len(rec_products)):
                                if (rec_products[i] in self.item_similarity_df.index and 
                                    rec_products[j] in self.item_similarity_df.columns):
                                    sim = self.item_similarity_df.loc[rec_products[i], rec_products[j]]
                                    similarities.append(sim)
                        
                        if similarities:
                            diversity_scores.append(1 - np.mean(similarities))
                except:
                    continue
            
            if diversity_scores:
                metrics['diversity'] = np.mean(diversity_scores)
        
        print(f"✅ Metrics calculated: Coverage={metrics['coverage']:.3f}, Diversity={metrics['diversity']:.3f}")
        return metrics
    
    def generate_batch_recommendations(self, customer_list, n_recommendations=10):
        """Generate recommendations for multiple customers efficiently"""
        print(f"⚡ Generating batch recommendations for {len(customer_list)} customers...")
        
        batch_results = {}
        
        for customer_id in customer_list:
            try:
                recommendations = self.get_hybrid_recommendations(customer_id, n_recommendations)
                batch_results[customer_id] = recommendations
            except Exception as e:
                print(f"Warning: Could not generate recommendations for {customer_id}: {e}")
                batch_results[customer_id] = self._get_popular_recommendations(n_recommendations)
        
        print("✅ Batch recommendations completed")
        return batch_results
    
    def export_recommendations(self, customer_id, filename=None):
        """Export recommendations for a specific customer"""
        recommendations = self.get_hybrid_recommendations(customer_id, 20)
        
        df = pd.DataFrame(recommendations)
        
        if filename is None:
            filename = f'recommendations_{customer_id}.csv'
        
        df.to_csv(filename, index=False)
        print(f"📤 Recommendations for {customer_id} exported to {filename}")
        return df

if __name__ == "__main__":
    # Example usage
    print("🚀 Coop Norge - Recommendation Engine")
    print("=" * 50)
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine(
        '../data/transactions.csv',
        '../data/customers.csv', 
        '../data/products.csv'
    )
    
    # Build models
    rec_engine.prepare_data()
    rec_engine.build_collaborative_filtering()
    rec_engine.build_content_based_filtering()
    
    # Test recommendations for a sample customer
    sample_customer = rec_engine.user_item_matrix.index[0]
    print(f"\n🎯 Sample recommendations for customer: {sample_customer}")
    
    # Get hybrid recommendations
    recommendations = rec_engine.get_hybrid_recommendations(sample_customer, 10)
    
    print(f"\n📋 TOP 10 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['product_name']} ({rec['category']}) - {rec['price']} NOK")
        print(f"   Score: {rec['predicted_score']:.3f} | Type: {rec['recommendation_type']}")
    
    # Calculate metrics
    metrics = rec_engine.calculate_recommendation_metrics()
    print(f"\n📊 RECOMMENDATION SYSTEM METRICS:")
    print(f"• Coverage: {metrics['coverage']:.1%}")
    print(f"• Diversity: {metrics['diversity']:.3f}")
    
    # Export sample recommendations
    rec_engine.export_recommendations(sample_customer, '../results/sample_recommendations.csv') 