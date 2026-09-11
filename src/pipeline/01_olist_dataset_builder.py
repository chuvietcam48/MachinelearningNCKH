import pandas as pd
import numpy as np
from pathlib import Path
import json

def construct_olist_cohort():
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = repo_root / "data/raw/olist"
    out_dir = repo_root / "data/processed/olist"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Olist raw tables...")
    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    reviews = pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv")
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    
    time_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for c in time_cols:
        orders[c] = pd.to_datetime(orders[c])
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
    
    orders = orders[orders['order_status'] == 'delivered'].copy()
    df = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    
    reviews_valid = reviews[reviews['review_comment_message'].notna() & (reviews['review_comment_message'].str.strip() != '')]
    reviews_dedup = reviews_valid.sort_values('review_creation_date').groupby('order_id').first().reset_index()
    
    df_reviews = df.merge(reviews_dedup[['order_id', 'review_comment_message', 'review_creation_date', 'review_score', 'review_id']], on='order_id', how='inner')
    
    origins = df_reviews.groupby('customer_unique_id').first().reset_index()
    origins.rename(columns={
        'order_id': 'origin_order_id', 
        'order_purchase_timestamp': 'origin_purchase_time',
        'review_creation_date': 'origin_review_date',
        'review_comment_message': 'origin_review_text',
        'review_score': 'origin_review_score'
    }, inplace=True)
                            
    origins['origin_delivery_days'] = (origins['order_delivered_customer_date'] - origins['origin_purchase_time']).dt.total_seconds() / 86400
    
    origin_lookup = origins[['customer_unique_id', 'origin_order_id', 'origin_review_date']].copy()
    df_all_orders = df.merge(origin_lookup, on='customer_unique_id', how='inner')
    
    next_orders = df_all_orders[
        (df_all_orders['order_id'] != df_all_orders['origin_order_id']) & 
        (df_all_orders['order_purchase_timestamp'] > df_all_orders['origin_review_date'])
    ]
    
    first_next_orders = next_orders.sort_values('order_purchase_timestamp').groupby('customer_unique_id').first().reset_index()
    first_next_orders.rename(columns={'order_purchase_timestamp': 'next_purchase_time'}, inplace=True)
    
    dataset = origins.merge(first_next_orders[['customer_unique_id', 'next_purchase_time']], on='customer_unique_id', how='left')
    dataset['has_event'] = dataset['next_purchase_time'].notna().astype(int)
    
    max_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    dataset['duration_days'] = np.where(
        dataset['has_event'] == 1,
        (dataset['next_purchase_time'] - dataset['origin_review_date']).dt.total_seconds() / 86400,
        (max_date - dataset['origin_review_date']).dt.total_seconds() / 86400
    )
    dataset = dataset[dataset['duration_days'] > 0]
    
    print(f"\n[Full Eligible Population]")
    print(f"Total N = {len(dataset):,}")
    print(f"Events = {dataset['has_event'].sum():,}")
    print(f"Event Rate = {dataset['has_event'].mean():.2%}")
    
    dataset = dataset.sort_values('origin_purchase_time').reset_index(drop=True)
    split_idx = int(len(dataset) * 0.8)
    
    full_train = dataset.iloc[:split_idx].copy()
    full_test = dataset.iloc[split_idx:].copy()
    
    print(f"\n[Full Chronological Split Statistics]")
    print(f"Train N: {len(full_train):,} | Train Events: {full_train['has_event'].sum():,}")
    print(f"Test N: {len(full_test):,} | Test Events: {full_test['has_event'].sum():,}")
    
    MAX_COHORT = 15000
    if len(dataset) > MAX_COHORT:
        print(f"\n[Applying Computationally Constrained Sampling: N = {MAX_COHORT}]")
        train_target = int(MAX_COHORT * 0.8)
        test_target = MAX_COHORT - train_target
        
        train_cohort = full_train.sample(n=train_target, random_state=42).copy()
        test_cohort = full_test.sample(n=test_target, random_state=42).copy()
    else:
        train_cohort = full_train.copy()
        test_cohort = full_test.copy()
        
    print(f"\n[Analytical Cohort (Proportional Sample)]")
    print(f"Train Customers: {len(train_cohort):,} | Train Events: {train_cohort['has_event'].sum():,}")
    print(f"Test Customers: {len(test_cohort):,} | Test Events: {test_cohort['has_event'].sum():,}")
    
    items_agg = items.groupby('order_id').agg(
        origin_item_count=('order_item_id', 'max'),
        origin_price_sum=('price', 'sum'),
        origin_freight_sum=('freight_value', 'sum')
    ).reset_index()
    
    train_temp = train_cohort.merge(items_agg, left_on='origin_order_id', right_on='order_id', how='left')
    train_temp['origin_delivery_days'] = (train_temp['order_delivered_customer_date'] - train_temp['origin_purchase_time']).dt.total_seconds() / 86400
    
    train_median_delivery_days = train_temp['origin_delivery_days'].median()
    train_median_review_score = train_temp['origin_review_score'].median()
    
    def enrich_features(split_df):
        res = split_df.merge(items_agg, left_on='origin_order_id', right_on='order_id', how='left')
        res['origin_delivery_days'] = (res['order_delivered_customer_date'] - res['origin_purchase_time']).dt.total_seconds() / 86400
        res['origin_price_sum'] = res['origin_price_sum'].fillna(0)
        res['origin_freight_sum'] = res['origin_freight_sum'].fillna(0)
        res['origin_item_count'] = res['origin_item_count'].fillna(1)
        res['origin_delivery_days'] = res['origin_delivery_days'].fillna(train_median_delivery_days)
        res['origin_review_score'] = res['origin_review_score'].fillna(train_median_review_score)
        return res
        
    print("\nConstructing Features independently...")
    train_feat = enrich_features(train_cohort)
    test_feat = enrich_features(test_cohort)
    
    cols_to_export = [
        'customer_unique_id', 'origin_order_id', 'review_id', 'origin_purchase_time', 'origin_review_date',
        'has_event', 'duration_days', 'origin_review_text', 'origin_review_score',
        'origin_item_count', 'origin_price_sum', 'origin_freight_sum', 'origin_delivery_days'
    ]
    
    train_final = train_feat[cols_to_export].copy()
    test_final = test_feat[cols_to_export].copy()
    
    train_final.to_csv(out_dir / "olist_train_analytical.csv", index=False)
    test_final.to_csv(out_dir / "olist_test_analytical.csv", index=False)
    
    info = {
        "full_N": len(dataset),
        "full_events": int(dataset['has_event'].sum()),
        "full_event_rate": float(dataset['has_event'].mean()),
        "analytical_train_size": len(train_final),
        "analytical_train_events": int(train_final['has_event'].sum()),
        "analytical_test_size": len(test_final),
        "analytical_test_events": int(test_final['has_event'].sum())
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(info, f, indent=4)
        
    print("Phase 2 Dataset Construction completed successfully.")

if __name__ == "__main__":
    construct_olist_cohort()
