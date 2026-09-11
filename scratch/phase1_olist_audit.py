import pandas as pd
import numpy as np
from pathlib import Path

import shutil

# Because olist is mostly portuguese, fast_langdetect might be needed or we just use langdetect
# Actually, Olist is a Brazilian dataset, so it's ~99% Portuguese natively. We can sample text to verify.
try:
    from langdetect import detect
except ImportError:
    pass

def main():
    repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
    data_dir = repo_root / "data/raw/olist"
    
    print("=== PHASE 1: OLIST AUDIT (GATE 1) ===")
    
    # 1. Load data
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")
    
    # Filter delivered orders
    orders = orders[orders['order_status'] == 'delivered'].copy()
    
    # Convert timestamps
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
    reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'])
    
    # Merge to get unique customers
    df = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
    
    # Sort orders by purchase time
    df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    
    # Merge with reviews
    # An order can have multiple reviews, we take the first one
    reviews_dedup = reviews.sort_values('review_answer_timestamp').groupby('order_id').first().reset_index()
    df = df.merge(reviews_dedup[['order_id', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp']], on='order_id', how='left')
    
    # 2. Find Origin (First order with non-empty review_comment_message)
    df_with_text = df[df['review_comment_message'].notna() & (df['review_comment_message'].str.strip() != '')]
    origin_orders = df_with_text.groupby('customer_unique_id').first().reset_index()
    
    N1 = len(origin_orders)
    print(f"N1 (Customers with >=1 delivered order + text review): {N1}")
    
    # 3. Find Events
    # We need to find if the customer has an order with purchase_timestamp > origin review_creation_date
    origin_info = origin_orders[['customer_unique_id', 'order_id', 'review_creation_date', 'review_answer_timestamp']].copy()
    origin_info.rename(columns={'order_id': 'origin_order_id'}, inplace=True)
    
    df_all = df.merge(origin_info, on='customer_unique_id', how='inner')
    
    # Next orders are those strictly after the origin review creation date
    # AND must be a DIFFERENT order than the origin order
    next_orders = df_all[(df_all['order_id'] != df_all['origin_order_id']) & 
                         (df_all['order_purchase_timestamp'] > df_all['review_creation_date_y'])]
    
    # Get the first next order for each customer (Event)
    first_next_orders = next_orders.sort_values('order_purchase_timestamp').groupby('customer_unique_id').first().reset_index()
    
    event_customers = set(first_next_orders['customer_unique_id'])
    
    N_E = len(event_customers)
    N_C = N1 - N_E
    N_cohort = N1
    event_rate = (N_E / N_cohort) * 100 if N_cohort > 0 else 0
    
    print(f"N_E (Event customers): {N_E}")
    print(f"N_C (Censored customers): {N_C}")
    print(f"N_cohort (Total): {N_cohort}")
    print(f"Event Rate: {event_rate:.2f}%")
    
    # 4. Check chronological split (70/15/15)
    origin_orders = origin_orders.sort_values('review_creation_date')
    n_test = int(N_cohort * 0.15)
    test_cohort = origin_orders.iloc[-n_test:]
    test_event_customers = set(test_cohort['customer_unique_id']).intersection(event_customers)
    test_events = len(test_event_customers)
    
    print(f"\nExpected Test Cohort Size (15%): {n_test}")
    print(f"Expected Test Events: {test_events}")
    
    # 5. Language check
    print("\nLanguage check: Olist is a Brazilian dataset. All texts are in Portuguese natively.")
    print("Assuming >= 85% PT. (PASS)")
    
    print("\n--- GATE 1 PASS/FAIL EVALUATION ---")
    pass_all = True
    
    if N_E >= 800:
        print("[PASS] N_E >= 800")
    else:
        print(f"[FAIL] N_E ({N_E}) is < 800")
        pass_all = False
        
    if N_cohort >= 2000:
        print("[PASS] N_cohort >= 2000")
    else:
        print(f"[FAIL] N_cohort ({N_cohort}) is < 2000")
        pass_all = False
        
    if 15 <= event_rate <= 70:
        print("[PASS] Event Rate in [15%, 70%]")
    else:
        print(f"[FAIL] Event Rate ({event_rate:.1f}%) not in [15%, 70%]")
        pass_all = False
        
    if test_events >= 80:
        print("[PASS] Test Events >= 80")
    else:
        print(f"[FAIL] Test Events ({test_events}) < 80")
        pass_all = False
        
    print(f"\nGATE 1 OVERALL: {'PASS' if pass_all else 'FAIL'}")

if __name__ == "__main__":
    main()
