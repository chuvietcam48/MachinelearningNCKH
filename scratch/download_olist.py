import kagglehub
import shutil
from pathlib import Path
import os
import pandas as pd

def main():
    print("Downloading Olist dataset via kagglehub...")
    # Download dataset
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print("Downloaded to:", path)
    
    # Destination directory
    repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
    dest_dir = repo_root / "data" / "raw" / "olist"
    
    # Move files to our data/raw/olist folder
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
        
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = dest_dir / item
        if not d.exists():
            shutil.copy2(s, d)
            
    print(f"Dataset successfully copied to: {dest_dir}")
    
    # Analyze repeat purchase rate
    print("\n--- Analyzing Olist Customer Repeat Rate ---")
    orders_path = dest_dir / "olist_orders_dataset.csv"
    if orders_path.exists():
        orders = pd.read_csv(orders_path)
        print(f"Total Orders: {len(orders)}")
        
        # Count orders per customer_unique_id
        # In Olist, 'customer_id' is order-specific, 'customer_unique_id' is the real user
        cust_path = dest_dir / "olist_customers_dataset.csv"
        customers = pd.read_csv(cust_path)
        
        df = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
        
        order_counts = df['customer_unique_id'].value_counts()
        total_unique_customers = len(order_counts)
        repeat_customers = (order_counts >= 2).sum()
        repeat_rate = (repeat_customers / total_unique_customers) * 100
        
        print(f"Total Unique Customers: {total_unique_customers}")
        print(f"Customers with >= 2 orders: {repeat_customers}")
        print(f"Repeat Purchase Rate: {repeat_rate:.2f}%")
        
        print("\nDistribution of Order Counts:")
        print(order_counts.value_counts().head(5))
        
        # How many have >= 2 orders AND wrote a review text on the first order?
        reviews_path = dest_dir / "olist_order_reviews_dataset.csv"
        reviews = pd.read_csv(reviews_path)
        
        df = df.merge(reviews[['order_id', 'review_comment_message']], on='order_id', how='left')
        
        # Sort by purchase time
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
        
        # Get first order for repeat customers
        repeat_cust_ids = order_counts[order_counts >= 2].index
        repeat_orders = df[df['customer_unique_id'].isin(repeat_cust_ids)]
        
        first_orders = repeat_orders.groupby('customer_unique_id').first()
        has_text_count = first_orders['review_comment_message'].notna().sum()
        
        print(f"\nOf the {repeat_customers} repeat customers, only {has_text_count} wrote a text review on their FIRST order.")
        print("This means the max possible cohort size for our lag-feedback framework is:", has_text_count)
        
    else:
        print("Orders dataset not found.")

if __name__ == "__main__":
    main()
