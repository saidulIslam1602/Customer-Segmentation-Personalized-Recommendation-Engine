"""
Dataset Generation for Customer Segmentation & Recommendation Engine
Generates realistic synthetic data that mirrors Coop Norge's data ecosystem
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker(['no_NO'])  # Norwegian locale
np.random.seed(42)
random.seed(42)

def generate_customer_data(n_customers=10000):
    """Generate customer/member data with Norwegian demographics"""
    customers = []
    
    for i in range(n_customers):
        # Age distribution reflecting Norwegian demographics
        age = np.random.normal(45, 15)
        age = max(18, min(80, int(age)))
        
        customer = {
            'customer_id': f'COOP_{i+1:06d}',
            'member_since': fake.date_between(start_date='-10y', end_date='today'),
            'age': age,
            'gender': np.random.choice(['M', 'F'], p=[0.49, 0.51]),
            'postal_code': fake.postcode(),
            'city': fake.city(),
            'county': np.random.choice(['Oslo', 'Bergen', 'Trondheim', 'Stavanger', 'Kristiansand', 
                                     'Fredrikstad', 'Sandnes', 'Tromsø', 'Sarpsborg', 'Skien']),
            'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                           p=[0.4, 0.3, 0.2, 0.1]),
            'email_consent': np.random.choice([True, False], p=[0.7, 0.3]),
            'mobile_app_user': np.random.choice([True, False], p=[0.6, 0.4]),
            'preferred_language': np.random.choice(['NO', 'EN', 'SE'], p=[0.85, 0.10, 0.05]),
            'household_size': np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.2, 0.2, 0.1]),
            'estimated_income': np.random.normal(550000, 150000)  # NOK
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_product_data(n_products=5000):
    """Generate product catalog data reflecting Norwegian retail"""
    categories = {
        'Dagligvarer': ['Brød og bakervarer', 'Meieri og egg', 'Kjøtt og fisk', 'Frukt og grønt', 
                       'Frosne varer', 'Tørrvarer', 'Drikke', 'Snacks og godteri'],
        'Hjem og fritid': ['Rengjøring', 'Tekstil', 'Elektronikk', 'Sport og fritid'],
        'Helse og skjønnhet': ['Apotek', 'Kosmetikk', 'Personlig pleie'],
        'Sesongvarer': ['Jul', 'Påske', 'Sommer', 'Tilbake til skolen']
    }
    
    products = []
    product_names = {
        'Brød og bakervarer': ['Rundstykker', 'Grovbrød', 'Croissant', 'Wienerbrød', 'Kringle'],
        'Meieri og egg': ['Helmelk', 'Lettmelk', 'Smør', 'Ost', 'Egg', 'Yoghurt', 'Rømme'],
        'Kjøtt og fisk': ['Kjøttdeig', 'Kyllingfilet', 'Laks', 'Torsk', 'Bacon', 'Pølser'],
        'Frukt og grønt': ['Bananer', 'Epler', 'Poteter', 'Gulrøtter', 'Tomater', 'Salat'],
        'Drikke': ['Coca Cola', 'Mineralvann', 'Kaffe', 'Te', 'Øl', 'Vin']
    }
    
    for i in range(n_products):
        main_cat = np.random.choice(list(categories.keys()))
        sub_cat = np.random.choice(categories[main_cat])
        
        # Generate realistic Norwegian prices
        if main_cat == 'Dagligvarer':
            price = np.random.uniform(5, 200)  # NOK
        elif main_cat == 'Hjem og fritid':
            price = np.random.uniform(20, 1500)
        else:
            price = np.random.uniform(10, 500)
        
        product = {
            'product_id': f'PROD_{i+1:06d}',
            'product_name': fake.word().capitalize() + ' ' + fake.word().capitalize(),
            'category': main_cat,
            'subcategory': sub_cat,
            'price': round(price, 2),
            'brand': fake.company(),
            'organic': np.random.choice([True, False], p=[0.15, 0.85]),
            'local_producer': np.random.choice([True, False], p=[0.25, 0.75]),
            'seasonal': sub_cat == 'Sesongvarer',
            'weight_kg': np.random.uniform(0.1, 5.0) if main_cat == 'Dagligvarer' else None,
            'launch_date': fake.date_between(start_date='-5y', end_date='today')
        }
        products.append(product)
    
    return pd.DataFrame(products)

def generate_transaction_data(customers_df, products_df, n_transactions=100000):
    """Generate transactional data - MOST CRITICAL for segmentation and recommendations"""
    transactions = []
    
    # Create customer behavior profiles
    customer_profiles = {}
    for _, customer in customers_df.iterrows():
        # Profile based on loyalty tier and demographics
        if customer['loyalty_tier'] == 'Platinum':
            freq_mult = np.random.normal(2.5, 0.3)  # Add variance
            spend_mult = np.random.normal(3.0, 0.4)
        elif customer['loyalty_tier'] == 'Gold':
            freq_mult = np.random.normal(2.0, 0.25)
            spend_mult = np.random.normal(2.0, 0.3)
        elif customer['loyalty_tier'] == 'Silver':
            freq_mult = np.random.normal(1.5, 0.2)
            spend_mult = np.random.normal(1.5, 0.25)
        else:
            freq_mult = np.random.normal(1.0, 0.15)
            spend_mult = np.random.normal(1.0, 0.2)
        
        # Ensure positive multipliers
        freq_mult = max(0.5, freq_mult)
        spend_mult = max(0.5, spend_mult)
        
        customer_profiles[customer['customer_id']] = {
            'frequency_multiplier': freq_mult,
            'spend_multiplier': spend_mult,
            'age_group': 'young' if customer['age'] < 35 else 'middle' if customer['age'] < 55 else 'senior',
            'household_size': customer['household_size']
        }
    
    transaction_id = 1
    for i in range(n_transactions):
        # Select customer with weighted probability (higher tier = more likely to shop)
        customer_weights = [customer_profiles[cid]['frequency_multiplier'] 
                          for cid in customers_df['customer_id']]
        customer_id = np.random.choice(customers_df['customer_id'], 
                                     p=np.array(customer_weights)/sum(customer_weights))
        
        profile = customer_profiles[customer_id]
        
        # Generate transaction date (more recent = higher probability)
        days_ago = int(np.random.exponential(30))  # Exponential distribution
        transaction_date = datetime.now() - timedelta(days=min(days_ago, 365))
        
        # Generate basket based on customer profile
        basket_size = max(1, int(np.random.poisson(3 * profile['household_size'])))
        
        # Select products based on age group preferences
        if profile['age_group'] == 'young':
            product_weights = [2 if 'Snacks' in str(p) or 'Drikke' in str(p) else 1 
                             for p in products_df['subcategory']]
        elif profile['age_group'] == 'senior':
            product_weights = [2 if 'Helse' in str(p) or 'Apotek' in str(p) else 1 
                             for p in products_df['category']]
        else:
            product_weights = [1] * len(products_df)
        
        product_weights = np.array(product_weights) / sum(product_weights)
        selected_products = np.random.choice(products_df.index, size=basket_size, 
                                           replace=False, p=product_weights)
        
        total_amount = 0
        for product_idx in selected_products:
            product = products_df.iloc[product_idx]
            quantity = max(1, int(np.random.poisson(1.5)))
            
            # Add realistic price variation (±10% from base price)
            price_variation = np.random.normal(1.0, 0.1)
            actual_price = product['price'] * max(0.5, price_variation)  # Ensure positive price
            
            item_total = actual_price * quantity * profile['spend_multiplier']
            
            # Add realistic spending noise
            spending_noise = np.random.normal(1.0, 0.05)  # ±5% spending variation
            item_total *= max(0.8, spending_noise)
            
            total_amount += item_total
            
            transaction = {
                'transaction_id': f'TXN_{transaction_id:08d}',
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'transaction_date': transaction_date.date(),
                'quantity': quantity,
                'unit_price': product['price'],
                'total_amount': round(item_total, 2),
                'store_id': f'STORE_{np.random.randint(1, 101):03d}',
                'channel': np.random.choice(['In-store', 'Online', 'Mobile App'], p=[0.7, 0.2, 0.1]),
                'payment_method': np.random.choice(['Card', 'Cash', 'Mobile Pay'], p=[0.6, 0.25, 0.15]),
                'discount_applied': np.random.choice([True, False], p=[0.3, 0.7]),
                'discount_amount': round(np.random.uniform(0, item_total * 0.2), 2) if np.random.random() < 0.3 else 0
            }
            transactions.append(transaction)
            transaction_id += 1
    
    return pd.DataFrame(transactions)

def generate_digital_events(customers_df, products_df, n_events=50000):
    """Generate digital channel events for personalization"""
    events = []
    
    event_types = ['page_view', 'product_view', 'add_to_cart', 'remove_from_cart', 
                   'search', 'email_open', 'email_click', 'app_open', 'push_notification_click']
    
    for i in range(n_events):
        customer_id = np.random.choice(customers_df['customer_id'])
        event_type = np.random.choice(event_types, p=[0.3, 0.25, 0.1, 0.05, 0.15, 0.05, 0.03, 0.05, 0.02])
        
        timestamp = fake.date_time_between(start_date='-6m', end_date='now')
        
        event = {
            'event_id': f'EVENT_{i+1:08d}',
            'customer_id': customer_id,
            'event_type': event_type,
            'timestamp': timestamp,
            'product_id': np.random.choice(products_df['product_id']) if event_type in ['product_view', 'add_to_cart', 'remove_from_cart'] else None,
            'session_id': f'SESSION_{np.random.randint(1, 10000):06d}',
            'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.4, 0.5, 0.1]),
            'channel': np.random.choice(['Website', 'Mobile App', 'Email'], p=[0.6, 0.3, 0.1]),
            'page_url': fake.url() if event_type == 'page_view' else None,
            'search_query': fake.word() if event_type == 'search' else None
        }
        events.append(event)
    
    return pd.DataFrame(events)

if __name__ == "__main__":
    print("Generating Customer Segmentation & Recommendation Engine Datasets...")
    print("=" * 60)
    
    # Generate datasets
    print("📊 Generating customer data...")
    customers_df = generate_customer_data(10000)
    customers_df.to_csv('customers.csv', index=False)
    print(f"✅ Generated {len(customers_df)} customer records")
    
    print("🛍️ Generating product catalog...")
    products_df = generate_product_data(5000)
    products_df.to_csv('products.csv', index=False)
    print(f"✅ Generated {len(products_df)} product records")
    
    print("💳 Generating transaction data (MOST CRITICAL)...")
    transactions_df = generate_transaction_data(customers_df, products_df, 100000)
    transactions_df.to_csv('transactions.csv', index=False)
    print(f"✅ Generated {len(transactions_df)} transaction records")
    
    print("📱 Generating digital events...")
    events_df = generate_digital_events(customers_df, products_df, 50000)
    events_df.to_csv('digital_events.csv', index=False)
    print(f"✅ Generated {len(events_df)} digital event records")
    
    print("\n" + "=" * 60)
    print("🎯 DATASET IMPACT RANKING FOR COOP NORGE ROLE:")
    print("1. 💳 TRANSACTIONS - Most critical for segmentation & recommendations")
    print("2. 👥 CUSTOMERS - Essential for demographic segmentation") 
    print("3. 📱 DIGITAL EVENTS - Key for personalization algorithms")
    print("4. 🛍️ PRODUCTS - Supporting data for content-based filtering")
    print("=" * 60)
    
    # Display sample data insights
    print(f"\n📈 SAMPLE INSIGHTS:")
    print(f"• Average transaction value: {transactions_df['total_amount'].mean():.2f} NOK")
    print(f"• Top spending customer tier: {transactions_df.groupby(customers_df.set_index('customer_id')['loyalty_tier'])['total_amount'].sum().idxmax()}")
    print(f"• Most popular product category: {products_df.merge(transactions_df, on='product_id')['category'].mode().iloc[0]}")
    print(f"• Digital engagement rate: {(events_df['event_type'] == 'email_open').sum() / len(events_df) * 100:.1f}%") 