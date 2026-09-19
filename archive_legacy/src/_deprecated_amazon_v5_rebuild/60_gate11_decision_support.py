import pandas as pd
import numpy as np
from pathlib import Path
import json

def get_action_and_cost(risk_level, value_level, reason):
    """
    Phase 11.3: Policy Routing Engine
    Maps Customer Risk, Value, and Semantic Reason to a prescriptive Action.
    """
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
            
    # Default for medium risk low value
    return "No Action Needed", 0.0, 0.0

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    gate11_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate11_policy"
    gate11_dir.mkdir(parents=True, exist_ok=True)
    
    print("Gate 11: Semantic-aware Intervention Policy Engine")
    print("=" * 60)
    
    # Load Predictions and Features
    try:
        pred_df = pd.read_parquet(gate10_dir / "prediction_model_c_semantic.parquet")
        feat_df = pd.read_parquet(gate10_dir / "master_test_semantic.parquet")
    except Exception as e:
        print(f"Required datasets not found: {e}")
        return
        
    # Merge, ignoring overlapping target columns in feat_df
    df = pd.merge(pred_df, feat_df.drop(columns=['E_Event', 'T_Duration'], errors='ignore'), on="episode_id")
    
    # Phase 11.1: Risk Stratification
    p50 = np.percentile(df['risk_score'], 50)
    p80 = np.percentile(df['risk_score'], 80)
    
    def get_risk(score):
        if score >= p80: return "High"
        if score >= p50: return "Medium"
        return "Low"
        
    df['Risk_Level'] = df['risk_score'].apply(get_risk)
    
    # Customer Value Stratification (Proxy: prior verified purchases + 1)
    df['CLV_Proxy'] = df['prior_verified_episode_count'] + 1
    val_p50 = np.percentile(df['CLV_Proxy'], 50)
    df['Value_Level'] = df['CLV_Proxy'].apply(lambda x: "High" if x >= val_p50 else "Low")
    
    # Phase 11.2: Semantic Risk Attribution
    def get_reason(row):
        if row.get('has_conflict', 0) == 1: return "Mixed"
        if row.get('Delivery_Packaging_Negative', 0) > 0: return "Delivery"
        if row.get('Price_Value_Negative', 0) > 0: return "Price"
        if row.get('Product_Performance_Usability_Negative', 0) > 0 or row.get('Quality_Reliability_Negative', 0) > 0: return "Product"
        if row.get('Customer_Service_Support_Negative', 0) > 0: return "Service"
        if row.get('rolling_negative_ratio', 0) > 0: return "Accumulated Negativity"
        return "Unknown"
        
    df['Semantic_Attribution'] = df.apply(get_reason, axis=1)
    
    # Phase 11.3: Policy Routing
    actions, costs, probs = [], [], []
    for _, row in df.iterrows():
        a, c, p = get_action_and_cost(row['Risk_Level'], row['Value_Level'], row['Semantic_Attribution'])
        actions.append(a)
        costs.append(c)
        probs.append(p)
        
    df['Recommended_Intervention'] = actions
    df['Intervention_Cost'] = costs
    df['Success_Prob'] = probs
    
    # Phase 11.4: Prioritization (ROI Calculation)
    # ROI Score = (Risk * CLV * SuccessProb) / Cost
    df['Priority_Score'] = (df['risk_score'] * df['CLV_Proxy'] * df['Success_Prob']) / (df['Intervention_Cost'] + 0.1) # smoothing
    
    # Sort and rank
    df_sorted = df.sort_values('Priority_Score', ascending=False)
    df_sorted[['episode_id', 'Risk_Level', 'Value_Level', 'Semantic_Attribution', 'Recommended_Intervention', 'Priority_Score', 'Intervention_Cost']].head(100).to_csv(gate11_dir / "priority_list_top100.csv", index=False)
    
    # Phase 11.5: Resource Allocation
    allocation = df.groupby('Recommended_Intervention').agg(
        Customers=('episode_id', 'count'),
        Total_Cost=('Intervention_Cost', 'sum'),
        Expected_ROI=('Priority_Score', 'mean')
    ).reset_index()
    allocation = allocation.sort_values('Total_Cost', ascending=False)
    allocation.to_csv(gate11_dir / "policy_routing_matrix.csv", index=False)
    
    # Phase 11.6: Counterfactual Policy Simulation
    # S(t) = exp(-cumulative_hazard). For simplification in this proxy simulation, we use Risk Score as relative hazard.
    # We assume baseline retention probability is derived from the actual event rate in the dataset.
    base_churn_rate = df['E_Event'].mean()
    base_retention = 1 - base_churn_rate
    
    # Scenarios: Effectiveness multiplier on the intervention success probability
    scenarios = {
        "Scenario A (Low Effectiveness)": 0.05,
        "Scenario B (Moderate Effectiveness)": 0.10,
        "Scenario C (High Effectiveness)": 0.20
    }
    
    counterfactual_results = []
    total_customers_intervened = len(df[df['Recommended_Intervention'] != "No Action Needed"])
    total_budget = df['Intervention_Cost'].sum()
    
    for sc_name, eff in scenarios.items():
        # Expected retained = Baseline Retained + Churned people saved
        # Saved = Base Churn Rate * (Intervention Effectiveness * Success Prob)
        saved_customers = df['E_Event'] * (eff * df['Success_Prob'])
        total_saved = saved_customers.sum()
        
        counterfactual_results.append({
            "Scenario": sc_name,
            "Baseline_Retained": len(df) - df['E_Event'].sum(),
            "Expected_Additional_Retained": total_saved,
            "Total_Retained_Post_Intervention": (len(df) - df['E_Event'].sum()) + total_saved,
            "Intervention_Budget": total_budget
        })
        
    cf_df = pd.DataFrame(counterfactual_results)
    cf_df.to_csv(gate11_dir / "counterfactual_simulation.csv", index=False)
    
    # Executive Summary Generation
    high_risk_pct = (len(df[df['Risk_Level'] == 'High']) / len(df)) * 100
    
    vouchers = len(df[df['Recommended_Intervention'] == 'Discount Voucher']) / len(df) * 100
    human = len(df[df['Recommended_Intervention'] == 'Human Escalation']) / len(df) * 100
    logistics = len(df[df['Recommended_Intervention'] == 'Agent + Logistics Recovery']) / len(df) * 100
    replacement = len(df[df['Recommended_Intervention'] == 'Replacement / Warranty']) / len(df) * 100
    generic = len(df[df['Recommended_Intervention'] == 'Generic Behavioral Campaign']) / len(df) * 100
    reminder = len(df[df['Recommended_Intervention'] == 'Retention Reminder']) / len(df) * 100
    no_action = len(df[df['Recommended_Intervention'] == 'No Action Needed']) / len(df) * 100
    
    exec_summary = f"""# Executive Decision Support Summary

## Top Findings
- **{high_risk_pct:.1f}%** of the evaluation cohort are classified as High-Risk based on survival hazard.
- **{vouchers:.1f}%** require Financial Interventions (Discount Vouchers) based on Price/Product attribution.
- **{logistics:.1f}%** require Logistics Recovery due to Delivery attribution.
- **{replacement:.1f}%** require Replacement/Warranty due to High-Value Product attribution.
- **{human:.1f}%** require Human Escalation due to Mixed/Conflicting semantic signals.
- **{generic:.1f}%** require Generic Behavioral Campaigns (High risk, no specific semantic reason).
- **{reminder:.1f}%** require Retention Reminders (Medium risk, High value).
- **{no_action:.1f}%** require No Action (Low risk or Medium risk & Low value).

## Financial & Operational Estimates
- **Total Intervention Budget Required:** ${total_budget:,.2f}
- **Targeted Customers (Non-Zero Action):** {total_customers_intervened}

## Expected Policy Benefit (Counterfactual)
| Scenario | Expected Additional Retained | Total Retained Post-Intervention |
|----------|------------------------------|----------------------------------|
| **Scenario A (Low)** | {cf_df.iloc[0]['Expected_Additional_Retained']:.1f} | {cf_df.iloc[0]['Total_Retained_Post_Intervention']:.1f} |
| **Scenario B (Moderate)** | {cf_df.iloc[1]['Expected_Additional_Retained']:.1f} | {cf_df.iloc[1]['Total_Retained_Post_Intervention']:.1f} |
| **Scenario C (High)** | {cf_df.iloc[2]['Expected_Additional_Retained']:.1f} | {cf_df.iloc[2]['Total_Retained_Post_Intervention']:.1f} |

*Note: Intervention success probabilities and effectiveness scenarios are hypothetical policy parameters used for counterfactual simulation rather than empirical causal estimates. The intervention policy is driven by semantic risk attribution rather than causal attribution.*
"""
    
    with open(gate11_dir / "executive_summary.md", "w") as f:
        f.write(exec_summary)
        
    print("\nGate 11 Execution Complete!")
    print("Files generated in: outputs/amazon_v5_rebuild/gate11_policy/")

if __name__ == "__main__":
    main()
