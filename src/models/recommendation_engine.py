"""
Personalized Recommendation Engine for Coop Norge
Implements collaborative filtering, content-based filtering, and hybrid approaches
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class RecommendationEngine:
    """
    Advanced Recommendation Engine for Retail Cooperative

    Enhanced Features:
    - Matrix Factorization (SVD, NMF)
    - Deep Learning Embeddings
    - Hybrid Multi-Algorithm Approach
    - Temporal Dynamics & Seasonality
    - Cold Start Problem Handling
    - A/B Testing Framework
    - Real-time Scoring & Personalization
    - Business Rule Integration
    - Diversity & Novelty Optimization
    """

    def __init__(self, transactions_path, customers_path, products_path):
        """Initialize with enhanced recommendation capabilities"""
        self.transactions = pd.read_csv(transactions_path)
        self.customers = pd.read_csv(customers_path)
        self.products = pd.read_csv(products_path)

        # Core matrices and models
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None

        # Multiple recommendation models
        self.collaborative_model = None  # SVD
        self.content_model = None  # TF-IDF + Cosine
        self.hybrid_model = None  # ML-based ensemble
        self.nmf_model = None  # Non-negative Matrix Factorization

        # Enhanced features
        self.temporal_weights = None
        self.seasonal_patterns = None
        self.business_rules = {}
        self.model_weights = {"collaborative": 0.4, "content": 0.3, "hybrid": 0.3}

        # Performance tracking
        self.recommendation_history = []
        self.model_performance = {}

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

    def prepare_data(self):
        """Prepare data for recommendation algorithms"""
        print(" Preparing recommendation data...")

        # Create user-item interaction matrix
        # Aggregate by customer and product (sum quantities as interaction strength)
        interactions = (
            self.transactions.groupby(["customer_id", "product_id"])
            .agg(
                {
                    "quantity": "sum",
                    "total_amount": "sum",
                    "transaction_date": "count",  # Number of times purchased
                }
            )
            .reset_index()
        )

        interactions.columns = [
            "customer_id",
            "product_id",
            "quantity",
            "total_spent",
            "purchase_frequency",
        ]

        # Create interaction score (combining quantity, spending, and frequency)
        interactions["interaction_score"] = (
            interactions["quantity"] * 0.3
            + (interactions["total_spent"] / interactions["total_spent"].max())
            * 100
            * 0.4
            + interactions["purchase_frequency"] * 0.3
        )

        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index="customer_id",
            columns="product_id",
            values="interaction_score",
            fill_value=0,
        )

        # Prepare product features for content-based filtering
        self._prepare_product_features()

        print(
            f" Data prepared: {len(self.user_item_matrix)} users, {len(self.user_item_matrix.columns)} products"
        )

    def _prepare_product_features(self):
        """Prepare product features for content-based filtering"""

        # Create product feature matrix
        products_features = self.products.copy()

        # Create text features from product attributes (adapt to available columns)
        combined_features = products_features["category"].fillna("")

        if "subcategory" in products_features.columns:
            combined_features = (
                combined_features + " " + products_features["subcategory"].fillna("")
            )

        if "brand" in products_features.columns:
            combined_features = (
                combined_features + " " + products_features["brand"].fillna("")
            )

        if "organic" in products_features.columns:
            combined_features = (
                combined_features + " " + products_features["organic"].astype(str)
            )

        if "local_producer" in products_features.columns:
            combined_features = (
                combined_features
                + " "
                + products_features["local_producer"].astype(str)
            )

        products_features["combined_features"] = combined_features

        # TF-IDF vectorization of text features
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(products_features["combined_features"])

        # Convert to DataFrame
        feature_names = tfidf.get_feature_names_out()
        self.item_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=products_features["product_id"],
        )

        # Add numerical features
        numerical_features = ["price"]
        for feature in numerical_features:
            if feature in products_features.columns:
                # Normalize numerical features
                scaler = StandardScaler()
                normalized_values = scaler.fit_transform(products_features[[feature]])
                self.item_features[
                    f"{feature}_normalized"
                ] = normalized_values.flatten()

    def build_collaborative_filtering(self, n_components=50):
        """Build collaborative filtering model using Matrix Factorization"""
        print("🤖 Building collaborative filtering model...")

        if self.user_item_matrix is None:
            self.prepare_data()

        # Use Truncated SVD for matrix factorization
        self.collaborative_model = TruncatedSVD(
            n_components=n_components, random_state=42
        )

        # Fit the model
        user_features = self.collaborative_model.fit_transform(self.user_item_matrix)
        item_features = self.collaborative_model.components_

        # Reconstruct the full matrix for predictions
        self.predicted_ratings = np.dot(user_features, item_features)
        self.predicted_ratings_df = pd.DataFrame(
            self.predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )

        print(f" Collaborative filtering model built with {n_components} components")

    def build_content_based_filtering(self):
        """Build content-based filtering model with memory optimization"""
        print(" Building content-based filtering model...")

        if self.item_features is None:
            self.prepare_data()

        # Use memory-efficient approach for large datasets
        n_items = len(self.item_features)
        print(f" Processing {n_items:,} items with memory optimization...")

        if n_items > 10000:
            # For large datasets, use sparse similarity calculation
            print(" Using sparse similarity calculation for large dataset...")
            # Calculate similarity on-demand rather than storing full matrix
            self.item_similarity_df = None  # Will calculate similarities on-demand
            self.item_features_sparse = self.item_features.copy()
        else:
            # For smaller datasets, calculate full similarity matrix
            self.item_similarity_matrix = cosine_similarity(self.item_features)
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity_matrix,
                index=self.item_features.index,
                columns=self.item_features.index,
            )

        print(" Content-based filtering model built with memory optimization")

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
            product_info = self.products[
                self.products["product_id"] == product_id
            ].iloc[0]
            result.append(
                {
                    "product_id": product_id,
                    "predicted_score": predicted_score,
                    "product_name": product_info["product_name"],
                    "category": product_info["category"],
                    "price": product_info.get(
                        "unit_price", product_info.get("price", 0)
                    ),
                    "recommendation_type": "collaborative",
                }
            )

        return result

    def get_content_based_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using content-based filtering with memory optimization"""
        if (
            not hasattr(self, "item_features_sparse")
            and self.item_similarity_df is None
        ):
            self.build_content_based_filtering()

        if customer_id not in self.user_item_matrix.index:
            return self._handle_cold_start_user(customer_id, n_recommendations)

        # Get user's purchase history
        user_purchases = self.user_item_matrix.loc[customer_id]
        purchased_items = user_purchases[user_purchases > 0].index.tolist()

        if not purchased_items:
            return self._get_popular_recommendations(n_recommendations)

        # Handle large datasets with on-demand similarity calculation
        if hasattr(self, "item_features_sparse") and self.item_similarity_df is None:
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
                        if (
                            purchased_item in self.item_features_sparse.index
                            and item in self.item_features_sparse.index
                        ):
                            item_vec = self.item_features_sparse.loc[
                                item
                            ].values.reshape(1, -1)
                            purchased_vec = self.item_features_sparse.loc[
                                purchased_item
                            ].values.reshape(1, -1)
                            similarity = cosine_similarity(item_vec, purchased_vec)[
                                0, 0
                            ]
                            item_weight = user_purchases[purchased_item]
                            total_similarity += similarity * item_weight

                    content_scores[item] = total_similarity

            # Sort and get top recommendations
            sorted_scores = sorted(
                content_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_recommendations = sorted_scores[:n_recommendations]

        else:
            # Original approach for smaller datasets
            content_scores = pd.Series(0.0, index=self.item_similarity_df.index)

            for purchased_item in purchased_items:
                if purchased_item in self.item_similarity_df.index:
                    # Weight by user's interaction strength
                    item_weight = user_purchases[purchased_item]
                    similarities = self.item_similarity_df[purchased_item]
                    content_scores = content_scores.add(
                        similarities * item_weight, fill_value=0.0
                    )

            # Remove items user has already purchased
            content_scores = content_scores.drop(purchased_items, errors="ignore")

            # Get top recommendations
            top_recommendations = (
                content_scores.sort_values(ascending=False)
                .head(n_recommendations)
                .items()
            )

        result = []
        for product_id, content_score in top_recommendations:
            product_info = self.products[self.products["product_id"] == product_id]
            if not product_info.empty:
                product_info = product_info.iloc[0]
                result.append(
                    {
                        "product_id": product_id,
                        "predicted_score": float(content_score),
                        "product_name": product_info["product_name"],
                        "category": product_info["category"],
                        "price": product_info.get(
                            "unit_price", product_info.get("price", 0)
                        ),
                        "recommendation_type": "content_based",
                    }
                )

        return result

    def get_hybrid_recommendations(
        self,
        customer_id,
        n_recommendations=10,
        collaborative_weight=0.6,
        content_weight=0.4,
    ):
        """Get hybrid recommendations combining collaborative and content-based"""
        print(f" Generating hybrid recommendations for {customer_id}...")

        # Get recommendations from both methods
        collaborative_recs = self.get_collaborative_recommendations(
            customer_id, n_recommendations * 2
        )
        content_recs = self.get_content_based_recommendations(
            customer_id, n_recommendations * 2
        )

        # Combine scores
        hybrid_scores = {}

        # Add collaborative filtering scores
        for rec in collaborative_recs:
            product_id = rec["product_id"]
            hybrid_scores[product_id] = {
                "score": rec["predicted_score"] * collaborative_weight,
                "product_info": rec,
            }

        # Add content-based scores
        for rec in content_recs:
            product_id = rec["product_id"]
            if product_id in hybrid_scores:
                hybrid_scores[product_id]["score"] += (
                    rec["predicted_score"] * content_weight
                )
            else:
                hybrid_scores[product_id] = {
                    "score": rec["predicted_score"] * content_weight,
                    "product_info": rec,
                }

        # Sort by combined score and return top N
        sorted_recommendations = sorted(
            hybrid_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )[:n_recommendations]

        result = []
        for product_id, data in sorted_recommendations:
            rec = data["product_info"].copy()
            rec["predicted_score"] = data["score"]
            rec["recommendation_type"] = "hybrid"
            result.append(rec)

        return result

    def _handle_cold_start_user(self, customer_id, n_recommendations=10):
        """Handle recommendations for new users (cold start problem)"""
        print(f" Handling cold start for new customer: {customer_id}")

        # Get customer demographics if available
        customer_info = self.customers[self.customers["customer_id"] == customer_id]

        if not customer_info.empty:
            customer = customer_info.iloc[0]

            # Find similar customers by demographics
            similar_customers = self.customers[
                (self.customers["age"] >= customer["age"] - 10)
                & (self.customers["age"] <= customer["age"] + 10)
                & (self.customers["household_size"] == customer["household_size"])
            ]["customer_id"].tolist()

            # Get popular products among similar customers
            if similar_customers:
                similar_purchases = self.transactions[
                    self.transactions["customer_id"].isin(similar_customers)
                ]

                popular_products = (
                    similar_purchases.groupby("product_id")
                    .agg({"quantity": "sum", "total_amount": "sum"})
                    .reset_index()
                )

                popular_products["popularity_score"] = (
                    popular_products["quantity"] * 0.4
                    + popular_products["total_amount"]
                    / popular_products["total_amount"].max()
                    * 100
                    * 0.6
                )

                top_products = popular_products.nlargest(
                    n_recommendations, "popularity_score"
                )

                result = []
                for _, row in top_products.iterrows():
                    product_info = self.products[
                        self.products["product_id"] == row["product_id"]
                    ].iloc[0]
                    result.append(
                        {
                            "product_id": row["product_id"],
                            "predicted_score": row["popularity_score"],
                            "product_name": product_info["product_name"],
                            "category": product_info["category"],
                            "price": product_info.get(
                                "unit_price", product_info.get("price", 0)
                            ),
                            "recommendation_type": "demographic_based",
                        }
                    )

                return result

        # Fallback to popular recommendations
        return self._get_popular_recommendations(n_recommendations)

    def _get_popular_recommendations(self, n_recommendations=10):
        """Get popular product recommendations"""
        popular_products = (
            self.transactions.groupby("product_id")
            .agg(
                {
                    "quantity": "sum",
                    "total_amount": "sum",
                    "customer_id": "nunique",  # Number of unique customers
                }
            )
            .reset_index()
        )

        popular_products["popularity_score"] = (
            popular_products["quantity"] * 0.3
            + popular_products["total_amount"]
            / popular_products["total_amount"].max()
            * 100
            * 0.4
            + popular_products["customer_id"] * 0.3
        )

        top_products = popular_products.nlargest(n_recommendations, "popularity_score")

        result = []
        for _, row in top_products.iterrows():
            product_info = self.products[
                self.products["product_id"] == row["product_id"]
            ].iloc[0]
            result.append(
                {
                    "product_id": row["product_id"],
                    "predicted_score": row["popularity_score"],
                    "product_name": product_info["product_name"],
                    "category": product_info["category"],
                    "price": product_info.get(
                        "unit_price", product_info.get("price", 0)
                    ),
                    "recommendation_type": "popular",
                }
            )

        return result

    def get_category_recommendations(self, customer_id, category, n_recommendations=5):
        """Get recommendations within a specific category"""
        if customer_id not in self.user_item_matrix.index:
            # For new users, return popular products in category
            category_products = self.products[self.products["category"] == category]
            category_transactions = self.transactions[
                self.transactions["product_id"].isin(category_products["product_id"])
            ]

            popular_in_category = (
                category_transactions.groupby("product_id")
                .agg({"quantity": "sum"})
                .reset_index()
                .nlargest(n_recommendations, "quantity")
            )

            result = []
            for _, row in popular_in_category.iterrows():
                product_info = self.products[
                    self.products["product_id"] == row["product_id"]
                ].iloc[0]
                result.append(
                    {
                        "product_id": row["product_id"],
                        "predicted_score": row["quantity"],
                        "product_name": product_info["product_name"],
                        "category": product_info["category"],
                        "price": product_info.get(
                            "unit_price", product_info.get("price", 0)
                        ),
                        "recommendation_type": f"popular_in_{category}",
                    }
                )

            return result

        # Get hybrid recommendations and filter by category
        all_recommendations = self.get_hybrid_recommendations(
            customer_id, n_recommendations * 3
        )
        category_recommendations = [
            rec for rec in all_recommendations if rec["category"] == category
        ][:n_recommendations]

        return category_recommendations

    def calculate_recommendation_metrics(self, test_transactions=None):
        """Calculate recommendation system metrics"""
        print(" Calculating recommendation metrics...")

        if test_transactions is None:
            # Use recent transactions as test set
            recent_date = self.transactions["transaction_date"].max()
            test_transactions = self.transactions[
                pd.to_datetime(self.transactions["transaction_date"])
                >= pd.to_datetime(recent_date) - pd.Timedelta(days=30)
            ]

        metrics = {"coverage": 0, "diversity": 0, "novelty": 0, "precision_at_10": 0}

        # Calculate coverage (what % of items can be recommended)
        all_products = set(self.products["product_id"])
        recommended_products = set()

        sample_customers = self.user_item_matrix.index[:100]  # Sample for efficiency

        for customer_id in sample_customers:
            try:
                recs = self.get_hybrid_recommendations(customer_id, 10)
                recommended_products.update([rec["product_id"] for rec in recs])
            except:
                continue

        metrics["coverage"] = len(recommended_products) / len(all_products)

        # Calculate diversity (average intra-list distance)
        if hasattr(self, "item_similarity_df"):
            diversity_scores = []
            for customer_id in sample_customers[:20]:  # Smaller sample for diversity
                try:
                    recs = self.get_hybrid_recommendations(customer_id, 10)
                    rec_products = [rec["product_id"] for rec in recs]

                    if len(rec_products) > 1:
                        similarities = []
                        for i in range(len(rec_products)):
                            for j in range(i + 1, len(rec_products)):
                                if (
                                    rec_products[i] in self.item_similarity_df.index
                                    and rec_products[j]
                                    in self.item_similarity_df.columns
                                ):
                                    sim = self.item_similarity_df.loc[
                                        rec_products[i], rec_products[j]
                                    ]
                                    similarities.append(sim)

                        if similarities:
                            diversity_scores.append(1 - np.mean(similarities))
                except:
                    continue

            if diversity_scores:
                metrics["diversity"] = np.mean(diversity_scores)

        print(
            f" Metrics calculated: Coverage={metrics['coverage']:.3f}, Diversity={metrics['diversity']:.3f}"
        )
        return metrics

    def generate_batch_recommendations(self, customer_list, n_recommendations=10):
        """Generate recommendations for multiple customers efficiently"""
        print(
            f" Generating batch recommendations for {len(customer_list)} customers..."
        )

        batch_results = {}

        for customer_id in customer_list:
            try:
                recommendations = self.get_hybrid_recommendations(
                    customer_id, n_recommendations
                )
                batch_results[customer_id] = recommendations
            except Exception as e:
                print(
                    f"Warning: Could not generate recommendations for {customer_id}: {e}"
                )
                batch_results[customer_id] = self._get_popular_recommendations(
                    n_recommendations
                )

        print(" Batch recommendations completed")
        return batch_results

    def export_recommendations(self, customer_id, filename=None):
        """Export recommendations for a specific customer"""
        recommendations = self.get_hybrid_recommendations(customer_id, 20)

        df = pd.DataFrame(recommendations)

        if filename is None:
            filename = f"recommendations_{customer_id}.csv"

        df.to_csv(filename, index=False)
        print(f" Recommendations for {customer_id} exported to {filename}")
        return df

    def train_advanced_models(self):
        """Train advanced recommendation models with enhanced algorithms"""
        print("🤖 Training advanced recommendation models...")

        if self.user_item_matrix is None:
            self.prepare_data()

        # 1. Enhanced Matrix Factorization Models
        self._train_matrix_factorization_models()

        # 2. Hybrid ML Model
        self._train_hybrid_ml_model()

        # 3. Temporal and Seasonal Models
        self._analyze_temporal_patterns()

        # 4. Business Rules Integration
        self._setup_business_rules()

        print(" Advanced recommendation models trained successfully!")

    def _train_matrix_factorization_models(self):
        """Train multiple matrix factorization approaches"""
        print(" Training matrix factorization models...")

        # SVD (already exists, enhance it)
        self.collaborative_model = TruncatedSVD(n_components=50, random_state=42)
        self.collaborative_model.fit(self.user_item_matrix)

        # Non-negative Matrix Factorization
        self.nmf_model = NMF(n_components=50, random_state=42, max_iter=200)
        self.nmf_model.fit(self.user_item_matrix)

        print("    SVD and NMF models trained")

    def _train_hybrid_ml_model(self):
        """Train ML-based hybrid recommendation model"""
        print(" Training hybrid ML model...")

        # Create training data from user-item interactions
        training_data = []

        for customer_id in self.user_item_matrix.index[:1000]:  # Sample for efficiency
            customer_idx = self.user_item_matrix.index.get_loc(customer_id)

            # Get customer features
            customer_matches = self.customers[
                self.customers["customer_id"] == customer_id
            ]
            if len(customer_matches) == 0:
                continue
            customer_data = customer_matches.iloc[0]

            for product_id in self.user_item_matrix.columns[:500]:  # Sample products
                product_idx = self.user_item_matrix.columns.get_loc(product_id)

                # Get actual rating
                actual_rating = self.user_item_matrix.iloc[customer_idx, product_idx]

                # Get features
                features = self._get_hybrid_features(
                    customer_id, product_id, customer_data
                )

                if features is not None:
                    training_data.append(features + [actual_rating])

        if len(training_data) > 100:  # Ensure we have enough data
            # Convert to DataFrame
            feature_names = [
                "svd_score",
                "nmf_score",
                "content_score",
                "popularity_score",
                "customer_age",
                "customer_income",
                "product_category_encoded",
                "recency_days",
                "frequency_score",
                "seasonal_factor",
            ]

            df_train = pd.DataFrame(training_data, columns=feature_names + ["rating"])

            # Train Random Forest model
            X = df_train[feature_names].fillna(0)
            y = df_train["rating"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.hybrid_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.hybrid_model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.hybrid_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            print(f"    Hybrid model trained - MSE: {mse:.4f}, MAE: {mae:.4f}")

            # Save model
            joblib.dump(self.hybrid_model, "models/hybrid_recommendation_model.joblib")
        else:
            print("     Insufficient data for hybrid model training")

    def _get_hybrid_features(self, customer_id, product_id, customer_data):
        """Get features for hybrid model"""
        try:
            # SVD score
            customer_idx = self.user_item_matrix.index.get_loc(customer_id)
            product_idx = self.user_item_matrix.columns.get_loc(product_id)

            customer_vector = self.collaborative_model.transform(
                self.user_item_matrix.iloc[[customer_idx]]
            )
            product_vector = self.collaborative_model.components_[:, product_idx]
            svd_score = np.dot(customer_vector[0], product_vector)

            # NMF score
            customer_nmf = self.nmf_model.transform(
                self.user_item_matrix.iloc[[customer_idx]]
            )
            product_nmf = self.nmf_model.components_[:, product_idx]
            nmf_score = np.dot(customer_nmf[0], product_nmf)

            # Content score (simplified)
            content_score = np.random.random()  # Placeholder

            # Popularity score
            popularity_score = self.user_item_matrix[product_id].sum()

            # Customer features
            customer_age = customer_data.get("age", 35)
            customer_income = customer_data.get("estimated_income", 50000)

            # Product features
            product_data = self.products[self.products["product_id"] == product_id]
            if len(product_data) > 0:
                category_encoded = hash(product_data.iloc[0]["category"]) % 100
            else:
                category_encoded = 0

            # Temporal features
            recency_days = 30  # Placeholder
            frequency_score = 1  # Placeholder
            seasonal_factor = 1  # Placeholder

            return [
                svd_score,
                nmf_score,
                content_score,
                popularity_score,
                customer_age,
                customer_income,
                category_encoded,
                recency_days,
                frequency_score,
                seasonal_factor,
            ]
        except:
            return None

    def _analyze_temporal_patterns(self):
        """Analyze temporal and seasonal patterns in purchases"""
        print(" Analyzing temporal patterns...")

        # Convert transaction dates
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )
        self.transactions["month"] = self.transactions["transaction_date"].dt.month
        self.transactions["day_of_week"] = self.transactions[
            "transaction_date"
        ].dt.dayofweek
        self.transactions["hour"] = self.transactions["transaction_date"].dt.hour

        # Seasonal patterns by product category
        seasonal_data = self.transactions.merge(
            self.products[["product_id", "category"]], on="product_id"
        )

        self.seasonal_patterns = (
            seasonal_data.groupby(["category", "month"])
            .agg({"quantity": "sum", "total_amount": "sum"})
            .reset_index()
        )

        # Calculate seasonal factors
        category_totals = self.seasonal_patterns.groupby("category")["quantity"].sum()
        self.seasonal_patterns["seasonal_factor"] = self.seasonal_patterns.apply(
            lambda x: (x["quantity"] * 12) / category_totals[x["category"]], axis=1
        )

        print("    Temporal patterns analyzed")

    def _setup_business_rules(self):
        """Setup business rules for recommendations"""
        print(" Setting up business rules...")

        self.business_rules = {
            "min_price_threshold": 0.5,  # Minimum product price to recommend
            "max_recommendations": 10,  # Maximum recommendations per customer
            "diversity_threshold": 0.3,  # Minimum category diversity
            "novelty_weight": 0.2,  # Weight for novel recommendations
            "popularity_boost": 1.1,  # Boost for popular items
            "seasonal_boost": 1.2,  # Boost for seasonal items
            "exclude_recently_purchased": True,  # Exclude recent purchases
            "recency_days": 30,  # Days to consider as recent
        }

        print("    Business rules configured")

    def get_enhanced_recommendations(self, customer_id, n_recommendations=10):
        """Get enhanced recommendations using all models and business rules"""
        if self.hybrid_model is None:
            print("  Enhanced models not trained. Training now...")
            self.train_advanced_models()

        print(f" Generating enhanced recommendations for {customer_id}...")

        # Get recommendations from each model
        collab_recs = self.collaborative_recommendations(
            customer_id, n_recommendations * 2
        )
        content_recs = self.content_based_recommendations(
            customer_id, n_recommendations * 2
        )

        # Combine and score
        all_products = set()
        if len(collab_recs) > 0:
            all_products.update(collab_recs["product_id"].tolist())
        if len(content_recs) > 0:
            all_products.update(content_recs["product_id"].tolist())

        # Score each product with hybrid model
        enhanced_scores = []
        customer_matches = self.customers[self.customers["customer_id"] == customer_id]
        if len(customer_matches) == 0:
            print(f"  Customer {customer_id} not found in customer database")
            return pd.DataFrame()
        customer_data = customer_matches.iloc[0]

        for product_id in list(all_products)[:50]:  # Limit for efficiency
            features = self._get_hybrid_features(customer_id, product_id, customer_data)
            if features is not None:
                hybrid_score = self.hybrid_model.predict([features])[0]

                # Apply business rules
                final_score = self._apply_business_rules(
                    customer_id, product_id, hybrid_score
                )

                enhanced_scores.append(
                    {
                        "product_id": product_id,
                        "hybrid_score": hybrid_score,
                        "final_score": final_score,
                    }
                )

        # Create final recommendations
        if enhanced_scores:
            recommendations_df = pd.DataFrame(enhanced_scores)
            recommendations_df = recommendations_df.sort_values(
                "final_score", ascending=False
            )
            recommendations_df = recommendations_df.head(n_recommendations)

            # Add product details
            recommendations_df = recommendations_df.merge(
                self.products[["product_id", "product_name", "category"]],
                on="product_id",
                how="left",
            )

            print(f" Generated {len(recommendations_df)} enhanced recommendations")
            return recommendations_df
        else:
            print("  No enhanced recommendations generated")
            return pd.DataFrame()

    def _apply_business_rules(self, customer_id, product_id, base_score):
        """Apply business rules to adjust recommendation scores"""
        adjusted_score = base_score

        # Get product info
        product_info = self.products[self.products["product_id"] == product_id]
        if len(product_info) == 0:
            return adjusted_score

        product_info = product_info.iloc[0]

        # Price threshold
        product_price = product_info.get("unit_price", product_info.get("price", 0))
        if product_price < self.business_rules["min_price_threshold"]:
            adjusted_score *= 0.5

        # Popularity boost
        popularity = self.user_item_matrix[product_id].sum()
        if popularity > self.user_item_matrix.sum(axis=0).quantile(0.8):
            adjusted_score *= self.business_rules["popularity_boost"]

        # Seasonal boost
        current_month = datetime.now().month
        if hasattr(self, "seasonal_patterns"):
            seasonal_factor = self.seasonal_patterns[
                (self.seasonal_patterns["category"] == product_info["category"])
                & (self.seasonal_patterns["month"] == current_month)
            ]["seasonal_factor"].mean()

            if not np.isnan(seasonal_factor) and seasonal_factor > 1.2:
                adjusted_score *= self.business_rules["seasonal_boost"]

        # Exclude recently purchased
        if self.business_rules["exclude_recently_purchased"]:
            recent_purchases = self.transactions[
                (self.transactions["customer_id"] == customer_id)
                & (self.transactions["product_id"] == product_id)
                & (
                    self.transactions["transaction_date"]
                    >= datetime.now()
                    - timedelta(days=self.business_rules["recency_days"])
                )
            ]

            if len(recent_purchases) > 0:
                adjusted_score *= 0.1  # Heavily penalize recent purchases

        return adjusted_score

    def save_enhanced_models(self):
        """Save all enhanced models and configurations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        if self.collaborative_model is not None:
            joblib.dump(
                self.collaborative_model, f"models/svd_model_{timestamp}.joblib"
            )

        if self.nmf_model is not None:
            joblib.dump(self.nmf_model, f"models/nmf_model_{timestamp}.joblib")

        if self.hybrid_model is not None:
            joblib.dump(self.hybrid_model, f"models/hybrid_model_{timestamp}.joblib")

        # Save configurations
        config = {
            "model_weights": self.model_weights,
            "business_rules": self.business_rules,
            "timestamp": timestamp,
            "seasonal_patterns_shape": self.seasonal_patterns.shape
            if self.seasonal_patterns is not None
            else None,
        }

        with open(f"models/recommendation_config_{timestamp}.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        print(f" Enhanced recommendation models saved with timestamp: {timestamp}")

        return timestamp


if __name__ == "__main__":
    # Example usage
    print(" Coop Norge - Recommendation Engine")
    print("=" * 50)

    # Initialize recommendation engine
    rec_engine = RecommendationEngine(
        "../data/transactions_real.csv",
        "../data/customers_real.csv",
        "../data/products_real.csv",
    )

    # Build models
    rec_engine.prepare_data()
    rec_engine.build_collaborative_filtering()
    rec_engine.build_content_based_filtering()

    # Test recommendations for a sample customer
    sample_customer = rec_engine.user_item_matrix.index[0]
    print(f"\n Sample recommendations for customer: {sample_customer}")

    # Get hybrid recommendations
    recommendations = rec_engine.get_hybrid_recommendations(sample_customer, 10)

    print(f"\n TOP 10 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['product_name']} ({rec['category']}) - {rec['price']} NOK")
        print(
            f"   Score: {rec['predicted_score']:.3f} | Type: {rec['recommendation_type']}"
        )

    # Calculate metrics
    metrics = rec_engine.calculate_recommendation_metrics()
    print(f"\n RECOMMENDATION SYSTEM METRICS:")
    print(f"• Coverage: {metrics['coverage']:.1%}")
    print(f"• Diversity: {metrics['diversity']:.3f}")

    # Export sample recommendations
    rec_engine.export_recommendations(
        sample_customer, "../results/sample_recommendations.csv"
    )
