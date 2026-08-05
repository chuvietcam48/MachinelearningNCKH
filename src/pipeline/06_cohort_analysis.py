import pandas as pd
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("06: Cohort Analysis (Selection Bias Acknowledgment)")
    print("=" * 70)
    
    cohorts = {
        "Entire Population (Broader)": "master_train_full.parquet",
        "Exact Cohort (Semantic)": "master_train_semantic.parquet"
    }
    
    results = []
    
    for name, file_name in cohorts.items():
        file_path = dataset_dir / file_name
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        df = pd.read_parquet(file_path)
        
        num_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
        total_episodes = len(df)
        episodes_per_cust = df.groupby('CustomerID').size() if 'CustomerID' in df.columns else pd.Series([1]*len(df))
        
        median_ep = episodes_per_cust.median()
        max_ep = episodes_per_cust.max()
        
        censoring_rate = 1.0 - df['E_Event'].mean()
        
        results.append({
            "Cohort": name,
            "Number of Customers": num_customers,
            "Total Episodes": total_episodes,
            "Episodes/Customer (Mean)": round(total_episodes / num_customers, 2),
            "Episodes/Customer (Median)": median_ep,
            "Episodes/Customer (Max)": max_ep,
            "Censoring Rate": f"{censoring_rate:.2%}"
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "cohort_analysis.csv", index=False)
    print(df_res.to_markdown(index=False))
    print("\nSaved cohort analysis to results/cohort_analysis.csv")

if __name__ == "__main__":
    main()
