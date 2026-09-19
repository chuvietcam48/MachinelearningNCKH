import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Please install vaderSentiment: pip install vaderSentiment")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

def _score_sentences(text, sia):
    sents = re.split(r'[.!?;]+', str(text))
    sents = [s.strip() for s in sents if len(s.strip()) > 8]
    if not sents:
        return 0, 0, 0, 0.0
    
    pos, neg = 0, 0
    compound_sum = 0.0
    for s in sents:
        scores = sia.polarity_scores(s)
        if scores["compound"] >= 0.05:
            pos += 1
        elif scores["compound"] <= -0.05:
            neg += 1
        compound_sum += scores["compound"]
        
    return pos, neg, len(sents), (compound_sum / len(sents))

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "data"
    
    # Store all Gate 10 output in gate10_baseline directory
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    gate10_dir.mkdir(parents=True, exist_ok=True)
    
    print("Gate 10.0: Traditional Sentiment Generation & Freeze")
    print("-" * 50)
    
    membership_path = data_dir / "verified_episode_review_membership_v1.parquet"
    print(f"Loading raw review data from: {membership_path.name}")
    
    try:
        df = pd.read_parquet(membership_path)
    except FileNotFoundError:
        print(f"Error: Could not find {membership_path}")
        sys.exit(1)
        
    if "reviewText" not in df.columns:
        print(f"'reviewText' not found. Available columns: {df.columns.tolist()}")
        if "text" in df.columns:
            df["reviewText"] = df["text"]
            
    sia = SentimentIntensityAnalyzer()
    
    print(f"Computing VADER sentiment for {len(df):,} reviews...")
    tqdm.pandas()
    
    def process_review(text):
        pos, neg, total, comp = _score_sentences(text, sia)
        return pd.Series([pos, neg, total, comp], index=['pos_sents', 'neg_sents', 'total_sents', 'mean_compound'])
        
    res = df['reviewText'].progress_apply(process_review)
    df = pd.concat([df, res], axis=1)
    
    print("\nAggregating sentiment to episode level...")
    episode_sentiment = df.groupby('episode_id').agg({
        'pos_sents': 'sum',
        'neg_sents': 'sum',
        'total_sents': 'sum',
        'mean_compound': 'mean' # Mean of mean compounds for the episode
    }).reset_index()
    
    # Compute mixed metrics & ratios
    total = episode_sentiment['total_sents'].replace(0, np.nan)
    
    episode_sentiment['sentiment_mixed_intensity'] = np.minimum(episode_sentiment['pos_sents'], episode_sentiment['neg_sents']) / total
    episode_sentiment['sentiment_positive_ratio'] = episode_sentiment['pos_sents'] / total
    episode_sentiment['sentiment_negative_ratio'] = episode_sentiment['neg_sents'] / total
    
    # Fill NAs
    for col in ['sentiment_mixed_intensity', 'sentiment_positive_ratio', 'sentiment_negative_ratio']:
        episode_sentiment[col] = episode_sentiment[col].fillna(0)
    
    # Rename for final output schema
    episode_sentiment = episode_sentiment.rename(columns={
        'pos_sents': 'sentiment_positive_sentences',
        'neg_sents': 'sentiment_negative_sentences',
        'total_sents': 'sentiment_total_sentences',
        'mean_compound': 'sentiment_mean_compound'
    })
    
    out_path = gate10_dir / "episode_sentiment_features.parquet"
    print(f"\nSaving Traditional Sentiment features to {out_path}...")
    episode_sentiment.to_parquet(out_path, index=False)
    
    print("Gate 10.0 completed successfully. Traditional Sentiment Frozen.")

if __name__ == "__main__":
    main()
