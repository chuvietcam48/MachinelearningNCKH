import pandas as pd
import numpy as np
from pathlib import Path

def get_action_and_cost(risk_level, value_level, reason):
    if risk_level == "Low":
        return "No Action Needed", 0.0, 0.0
        
    if risk_level == "Medium" and value_level == "High":
        return "Retention Reminder", 1.0, 0.15
        
    if risk_level == "High":
        if reason == "Delivery":
            if value_level == "High":
                return "Agent + Logistics Recovery", 20.0, 0.55
            else:
                return "Shipping Coupon", 5.0, 0.45
        elif reason == "Price":
            return "Discount Voucher", 10.0, 0.40
        elif reason == "Product" and value_level == "High":
            return "Replacement / Warranty", 50.0, 0.70
        elif reason == "Product" and value_level == "Low":
            return "Discount Voucher", 10.0, 0.40
        elif reason == "Mixed":
            return "Human Escalation", 15.0, 0.65
        else:
            return "Generic Behavioral Campaign", 5.0, 0.20
            
    return "No Action Needed", 0.0, 0.0

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    pred_dir = repo_root / "outputs" / "pipeline_freeze" / "predictions"
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    policy_dir = repo_root / "outputs" / "pipeline_freeze" / "results" / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    
    print("05: Semantic-aware Intervention Policy Engine")
    print("=" * 60)
    
    try:
        pred_df = pd.read_parquet(pred_dir / "prediction_model_c_semantic.parquet")
        feat_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    except Exception as e:
        print(f"Required datasets not found: {e}")
        return
        
    df = pd.merge(pred_df, feat_df.drop(columns=['E_Event', 'T_Duration'], errors='ignore'), on="episode_id")
    
    # 1. Risk Stratification (High risk means LOW survival_prob -> leaving soon)
    # But wait, we want to intervene on those who are NOT interacting soon.
    # Event = next interaction. Hazard = probability of next interaction.
    # Therefore, a low hazard (high survival_prob) means prolonged inactivity (CHURN).
    # We want to intervene when prolonged inactivity risk is HIGH.
    # So "High" risk actually means HIGH survival_prob (S(t*) > p80)
    
    # We will stratify by survival_prob (prolonged inactivity risk)
    p50 = np.percentile(df['survival_prob'].dropna(), 50)
    p80 = np.percentile(df['survival_prob'].dropna(), 80)
    
    df['Risk_Level'] = df['survival_prob'].apply(lambda x: "High" if x >= p80 else ("Medium" if x >= p50 else "Low"))
    
    df['CLV_Proxy'] = df['prior_verified_episode_count'] + 1
    val_p50 = np.percentile(df['CLV_Proxy'], 50)
    df['Value_Level'] = df['CLV_Proxy'].apply(lambda x: "High" if x >= val_p50 else "Low")
    
    # 2. Semantic Risk Attribution
    def get_reason(row):
        if row.get('has_conflict', 0) == 1: return "Mixed"
        if row.get('Delivery_Packaging_Negative', 0) > 0: return "Delivery"
        if row.get('Price_Value_Negative', 0) > 0: return "Price"
        if row.get('Product_Performance_Usability_Negative', 0) > 0 or row.get('Quality_Reliability_Negative', 0) > 0: return "Product"
        if row.get('Customer_Service_Support_Negative', 0) > 0: return "Service"
        if row.get('rolling_negative_ratio', 0) > 0: return "Accumulated Negativity"
        return "Unknown"
        
    df['Semantic_Attribution'] = df.apply(get_reason, axis=1)
    
    # 3. Policy Recommendation
    actions, costs, probs = [], [], []
    for _, row in df.iterrows():
        a, c, p = get_action_and_cost(row['Risk_Level'], row['Value_Level'], row['Semantic_Attribution'])
        actions.append(a)
        costs.append(c)
        probs.append(p)
        
    df['Recommended_Intervention'] = actions
    df['Intervention_Cost'] = costs
    df['Success_Prob'] = probs
    
    # 4. Priority Ranking
    df['Priority_Score'] = (df['survival_prob'] * df['CLV_Proxy'] * df['Success_Prob']) / (df['Intervention_Cost'] + 0.1)
    
    df_sorted = df.sort_values('Priority_Score', ascending=False)
    df_sorted[['episode_id', 'Risk_Level', 'Value_Level', 'Semantic_Attribution', 'Recommended_Intervention', 'Priority_Score', 'Intervention_Cost']].head(100).to_csv(policy_dir / "priority_list_top100.csv", index=False)
    
    # 5. Budget Allocation
    allocation = df.groupby('Recommended_Intervention').agg(
        Customers=('episode_id', 'count'),
        Total_Cost=('Intervention_Cost', 'sum'),
        Expected_ROI=('Priority_Score', 'mean')
    ).reset_index().sort_values('Total_Cost', ascending=False)
    allocation.to_csv(policy_dir / "policy_routing_matrix.csv", index=False)
    
    # 6. Scenario Evaluation
    scenarios = {
        "Scenario A (Low Effectiveness)": 0.05,
        "Scenario B (Moderate Effectiveness)": 0.10,
        "Scenario C (High Effectiveness)": 0.20
    }
    
    counterfactual_results = []
    for sc_name, eff in scenarios.items():
        saved_customers = df['E_Event'] * (eff * df['Success_Prob'])
        total_saved = saved_customers.sum()
        counterfactual_results.append({
            "Scenario": sc_name,
            "Baseline_Retained": len(df) - df['E_Event'].sum(),
            "Expected_Additional_Retained": total_saved,
            "Total_Retained_Post_Intervention": (len(df) - df['E_Event'].sum()) + total_saved,
            "Intervention_Budget": df['Intervention_Cost'].sum()
        })
        
    pd.DataFrame(counterfactual_results).to_csv(policy_dir / "scenario_evaluation.csv", index=False)
    
    # 7. Decision Sensitivity Analysis
    print("Running Decision Sensitivity Analysis...")
    gammas = [0.3, 0.5, 0.7] # Effectiveness scale
    cost_scalars = [0.5, 1.0, 2.0] # Base cost scale
    value_scalars = [0.5, 1.0, 2.0] # Value scale
    
    sens_results = []
    for g in gammas:
        for cs in cost_scalars:
            for vs in value_scalars:
                scaled_cost = df['Intervention_Cost'] * cs
                scaled_clv = df['CLV_Proxy'] * vs * 100 
                success_prob = df['Success_Prob'] * g
                
                expected_saved_value = success_prob * scaled_clv * df['survival_prob']
                net_roi = expected_saved_value - scaled_cost
                
                intervene_mask = net_roi > 0
                total_cost = scaled_cost[intervene_mask].sum()
                total_net_roi = net_roi[intervene_mask].sum()
                num_interventions = intervene_mask.sum()
                
                sens_results.append({
                    "Gamma": g,
                    "Cost_Scalar": cs,
                    "Value_Scalar": vs,
                    "Num_Interventions": num_interventions,
                    "Total_Cost": total_cost,
                    "Total_Net_ROI": total_net_roi
                })
                
    pd.DataFrame(sens_results).to_csv(policy_dir / "sensitivity_analysis.csv", index=False)
    
    print("\nDecision Support Engine Executed Successfully.")
    print("Files saved to results/policy/")

if __name__ == "__main__":
    main()
