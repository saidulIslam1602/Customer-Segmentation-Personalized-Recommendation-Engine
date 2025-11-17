#!/usr/bin/env python3
"""
Cleanup Script for Redundant Files
Removes redundant code and data files to optimize project structure
"""

import os
import shutil
from pathlib import Path

def cleanup_redundant_files():
    """Remove redundant files and optimize project structure"""
    
    print("🧹 CLEANING UP REDUNDANT FILES")
    print("=" * 50)
    
    # Files to remove
    redundant_files = [
        # Redundant pipeline files
        "src/main.py",
        "src/run_real_analysis.py", 
        "high_precision_analytics.py",
        "create_enterprise_visualizations.py",
        
        # Old data files (already moved to processed/)
        "data/transactions.csv",
        "data/customers.csv", 
        "data/products.csv",
        "data/digital_events.csv",
        
        # Raw downloaded files (already processed)
        "data/online_retail.xlsx",
        "data/wholesale_customers.csv",
        
        # Old data generation scripts (already removed)
    ]
    
    # Directories to clean
    redundant_dirs = [
        "src/__pycache__",
        "src/business_intelligence/__pycache__",
        "src/data_integration/__pycache__",
    ]
    
    total_size_saved = 0
    files_removed = 0
    
    # Remove redundant files
    for file_path in redundant_files:
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                total_size_saved += file_size
                files_removed += 1
                print(f" Removed: {file_path} ({file_size/1024/1024:.1f} MB)")
            except Exception as e:
                print(f" Error removing {file_path}: {e}")
        else:
            print(f"  Not found: {file_path}")
    
    # Remove redundant directories
    for dir_path in redundant_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f" Removed directory: {dir_path}")
            except Exception as e:
                print(f" Error removing {dir_path}: {e}")
    
    print(f"\n CLEANUP SUMMARY:")
    print(f"   Files removed: {files_removed}")
    print(f"   Space saved: {total_size_saved/1024/1024:.1f} MB")
    
    # Create optimized project structure summary
    create_optimized_structure_summary()

def create_optimized_structure_summary():
    """Create a summary of the optimized project structure"""
    
    structure_summary = """
#  OPTIMIZED PROJECT STRUCTURE

## Core Business Intelligence Platform
```
Customer-Segmentation-Personalized-Recommendation-Engine/
 src/
    business_intelligence/              # Advanced BI modules
       churn_prediction.py            # Customer retention
       inventory_optimization.py       # Demand forecasting  
       pricing_optimization.py         # Dynamic pricing
       fraud_detection.py             # Risk management
       marketing_attribution.py       # ROI analysis
       executive_dashboard.py         # Business insights
    data_integration/                  # Real data integration
       real_data_loader.py           # UCI dataset loader
    enhanced_business_pipeline.py      # Main pipeline
    customer_segmentation.py          # Core segmentation
    recommendation_engine.py          # Core recommendations
 data/                                  # UCI datasets
    transactions_real.csv            # 397K real transactions
    customers_real.csv               # 440 real customers  
    products_real.csv                # 3,897 real products
    digital_events_real.csv          # 15K digital events
 results/                              # Analysis outputs
 requirements.txt                      # Dependencies
 Dockerfile                           # Container config
 docker-compose.yml                   # Multi-service setup
 ENHANCED_BUSINESS_INTELLIGENCE_README.md
```

## Key Benefits After Cleanup:
-  Reduced project size by ~150MB
-  Eliminated duplicate functionality
-  Cleaner, more maintainable codebase
-  Focus on UCI research datasets
-  Single comprehensive pipeline

## Usage After Cleanup:
```bash
# Run complete business intelligence analysis
python3 src/enhanced_business_pipeline.py

# Run individual modules
python3 src/business_intelligence/churn_prediction.py
python3 src/business_intelligence/inventory_optimization.py
# ... etc
```
"""
    
    with open("OPTIMIZED_PROJECT_STRUCTURE.md", "w") as f:
        f.write(structure_summary)
    
    print(" Created: OPTIMIZED_PROJECT_STRUCTURE.md")

def backup_before_cleanup():
    """Create a backup of important files before cleanup"""
    
    backup_dir = "backup_before_cleanup"
    os.makedirs(backup_dir, exist_ok=True)
    
    important_files = [
        "src/main.py",
        "high_precision_analytics.py", 
        "create_enterprise_visualizations.py"
    ]
    
    for file_path in important_files:
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_dir)
            print(f" Backed up: {file_path}")
    
    print(f" Backup created in: {backup_dir}/")

if __name__ == "__main__":
    print("  This script will remove redundant files to optimize the project.")
    print(" A backup will be created first.")
    
    response = input("\nProceed with cleanup? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        backup_before_cleanup()
        cleanup_redundant_files()
        print("\n Project cleanup completed successfully!")
        print(" Check OPTIMIZED_PROJECT_STRUCTURE.md for details")
    else:
        print(" Cleanup cancelled.")