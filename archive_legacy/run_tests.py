import sys
from tests.test_pipeline import *

def run_all():
    print("Running Integration Tests...")
    try:
        test_dataset_integrity()
        print("[OK] test_dataset_integrity passed")
        
        test_hazard_ratio_consistency()
        print("[OK] test_hazard_ratio_consistency passed")
        
        test_policy_routing_validity()
        print("[OK] test_policy_routing_validity passed")
        
        test_sensitivity_results_exist()
        print("[OK] test_sensitivity_results_exist passed")
        
        test_experiment_registry()
        print("[OK] test_experiment_registry passed")
        
        print("\nAll integration tests passed successfully!")
    except Exception as e:
        print(f"\n[FAILED] Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all()
