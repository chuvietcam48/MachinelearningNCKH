import pandas as pd

x5_path = 'data/raw/x5retail/purchases.csv'
print(f"Reading {x5_path}...")
df = pd.read_csv(x5_path, usecols=['client_id', 'transaction_datetime'])
df['date'] = pd.to_datetime(df['transaction_datetime']).dt.date
max_date = df['date'].max()
snap = max_date + pd.Timedelta(days=1)
print('Dataset max date:', max_date, 'snapshot:', snap)
last_purchases = df.groupby('client_id')['date'].max()
recency = (snap - last_purchases).dt.days

print('\nTotal customers:', len(recency))
print('Recency distribution:')
print(recency.describe())

for t in [26, 39, 60, 90]:
    print(f'Customers with Recency > {t} days: {(recency > t).sum()}')
