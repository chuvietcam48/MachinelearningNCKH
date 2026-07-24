import argparse
import logging
from src.framework.dataset_adapter import DatasetAdapter
from src.framework.behavior_engine import BehaviorEngine
from src.framework.semantic_engine import SemanticEngine
from src.framework.feature_fusion import FeatureFusionEngine
from src.framework.survival_engine import SurvivalEngine
from src.framework.decision_support_engine import DecisionSupportEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Universal Customer Churn Decision Framework")
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset (e.g., cdnow, amazon, tafeng)")
    parser.add_argument("--enable-semantic", action="store_true", help="Activate optional Semantic Extension Module")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(" UNIVERSAL CUSTOMER CHURN DECISION FRAMEWORK")
    print("="*70)

    # 1. Dataset Adapter
    adapter = DatasetAdapter(dataset_name=args.dataset)
    df_raw = adapter.load_data()
    print(f"[Adapter] Loaded {adapter.get_display_name()} | Rows: {len(df_raw)}")
    
    # Get snapshot date
    from src.dataset_registry import get_dataset
    registry_info = get_dataset(args.dataset)
    snapshot_date = registry_info.snapshot_fn(df_raw)

    # 2. Behavior Engine
    behavior_engine = BehaviorEngine(snapshot_date=snapshot_date)
    df_behavior = behavior_engine.process(df_raw)
    print(f"[Behavior] Engineered base RFM and behavioral signals.")

    # 3. Plugin Manager & Optional Features
    from src.framework.plugin_manager import PluginManager
    plugin_manager = PluginManager(dataset_name=args.dataset, force_semantic=args.enable_semantic)
    semantic_provider = plugin_manager.get_semantic_provider()
    
    if semantic_provider:
        df_semantic_raw = semantic_provider.load()
        if len(df_semantic_raw) > 0:
            semantic_engine = SemanticEngine()
            df_semantic = semantic_engine.process(df_semantic_raw)
            
            fusion_engine = FeatureFusionEngine()
            df_master = fusion_engine.merge(df_behavior, df_semantic)
            print(f"[Semantic] Plugin activated. Injected aspect trajectory features (Validation Case).")
        else:
            df_master = df_behavior
            print(f"[Semantic] Plugin activated but no data loaded. Running on behavior only.")
    else:
        df_master = df_behavior
        if args.enable_semantic:
            logger.warning(f"Semantic flag passed, but '{args.dataset}' has no registered Semantic Provider.")
        print(f"[Semantic] Skipped. Running completely on behavioral signals.")

    # 4. Survival Engine
    survival_engine = SurvivalEngine(
        penalizer=adapter.config['survival_model']['penalizer'],
        l1_ratio=adapter.config['survival_model']['l1_ratio']
    )
    # Simple split for demonstration framework execution
    df_train = df_master.sample(frac=0.8, random_state=42).copy()
    df_test = df_master.drop(df_train.index).copy()
    
    # Needs valid duration and event columns to run CoxPH
    if 'T_Duration' in df_train.columns and 'E_Event' in df_train.columns:
        cph = survival_engine.fit(df_train)
        print(f"[Survival] Cox Proportional Hazards fitted. AIC: {cph.AIC_partial_:.2f}")
        
        # Predict risk for Decision Engine
        import numpy as np
        df_test_numeric = df_test.select_dtypes(include=[np.number])
        # Only keep columns that were actually fitted in the model
        valid_cols = [c for c in df_test_numeric.columns if c in cph.params_.index]
        risk_scores = cph.predict_partial_hazard(df_test_numeric[valid_cols])
        
        # 5. Decision Support Engine
        dss_engine = DecisionSupportEngine(
            high_risk_threshold=adapter.config['decision_support']['high_risk_threshold']
        )
        df_policy = dss_engine.route_customers(df_test, risk_scores)
        print(f"[DSS] Customer Routing Complete. Generated Policies:\n")
        print(df_policy['Recommended_Intervention'].value_counts())
    else:
        logger.error("Standard targets 'T_Duration' and 'E_Event' not found. Framework aborted.")

    print("\n" + "="*70)
    print(" Framework Execution Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
