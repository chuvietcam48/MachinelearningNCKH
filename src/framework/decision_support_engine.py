import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DecisionSupportEngine:
    """
    Layer 3: Decision Support Engine
    Provides adaptive routing based on Risk Attribution.
    If semantic features are present, routes via Aspect Conficts and Flip Rates.
    Otherwise, defaults to baseline risk percentile routing.
    """
    def __init__(self, high_risk_threshold=0.75):
        self.high_risk_threshold = high_risk_threshold

    def route_customers(self, df: pd.DataFrame, risk_scores: pd.Series) -> pd.DataFrame:
        """
        Dynamically applies routing rules based on available columns.
        """
        logger.info("Executing Decision Support Engine...")
        
        df_routing = pd.DataFrame({'CustomerID': df.get('CustomerID', df.index), 'Risk_Score': risk_scores})
        
        # Identify high-risk cohort based on empirical percentiles
        risk_cutoff = np.percentile(risk_scores.dropna(), self.high_risk_threshold * 100)
        df_routing['Is_High_Risk'] = df_routing['Risk_Score'] >= risk_cutoff
        
        policies = []
        has_semantic = 'cross_aspect_conflict' in df.columns or 'aspect_transition_pattern' in df.columns
        
        if has_semantic:
            logger.info("Semantic Extension active. Using Semantic Risk Attribution Routing.")
            for i, row in df.iterrows():
                if not df_routing.loc[i, 'Is_High_Risk']:
                    policies.append("No Action Needed")
                    continue
                    
                # Adaptive logic using semantic signals
                if row.get('cross_aspect_conflict', False):
                    policies.append("Immediate Service Recovery (Conflict Resolution)")
                elif row.get('aspect_transition_pattern', -1) > 0:
                    policies.append("Specialized Multi-department Follow-up")
                elif row.get('negative_streak', 0) >= 3:
                    policies.append("High-Priority Apology & Discount")
                else:
                    policies.append("Standard Retention Offer")
        else:
            logger.info("Using Default Behavioral Risk Routing.")
            for i, row in df.iterrows():
                if not df_routing.loc[i, 'Is_High_Risk']:
                    policies.append("No Action Needed")
                    continue
                
                # Baseline logic relying only on standard behavioral indicators
                if row.get('Recency', 999) < 7:
                    policies.append("Recent Churner: Rapid Discount Offer")
                elif row.get('Monetary', 0) > df['Monetary'].median():
                    policies.append("High Value Churner: VIP Account Manager Outreach")
                else:
                    policies.append("Standard Retention Email")
                    
        df_routing['Recommended_Intervention'] = policies
        return df_routing
