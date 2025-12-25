import joblib
import pandas as pd
import numpy as np
import shap
import os
from .schemas import PredictionInput

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../ml/artifacts/risk_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../ml/artifacts/label_encoder.joblib")

class RiskModel:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                # Initialize SHAP explainer
                # For Tree models like XGBoost, TreeExplainer is efficient
                self.explainer = shap.TreeExplainer(self.model)
                print("ML Model loaded successfully.")
            else:
                print("Warning: Model artifacts not found. Using fallback logic.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, input_data: PredictionInput):
        # Convert input to DataFrame (ordering matters!)
        features = pd.DataFrame([input_data.model_dump()])
        
        # Ensure column order matches training
        feature_order = ['Age', 'Annual_Income', 'Loan_Amount', 'Credit_Score', 'Employment_Years', 'Past_Defaults']
        features = features[feature_order]

        if self.model:
            # Prediction
            # XGBoost predict_proba returns [prob_class0, prob_class1, prob_class2]
            probs = self.model.predict_proba(features)[0]
            
            # Identify class
            pred_idx = np.argmax(probs)
            risk_level = self.encoder.classes_[pred_idx]
            
            # Calculate a risk probability (probability of being High Risk)
            # Assuming 'High' is one of the classes. 
            # We need to know which index is 'High'. 
            # If encoder.classes_ = ['High', 'Low', 'Medium'], then High is 0.
            
            high_risk_idx = np.where(self.encoder.classes_ == 'High')[0][0]
            risk_probability = float(probs[high_risk_idx])
            
            # Risk Score: 100 - (Risk Prob * 100). Higher is better/safer.
            # OR User wants Risk Score (0-100). Usually High Score = High Risk for "Risk Score"?
            # Let's assume Risk Score 0-100 means 100 is MAX RISK.
            risk_score = round(risk_probability * 100, 2)
            
            # SHAP Values
            shap_values = self.explainer.shap_values(features)
            
            # shap_values is a list of arrays for multiclass. We want the SHAP val for the predicted class or High Risk class.
            # Let's focus on factors driving 'High Risk'
            feature_shap = shap_values[features.shape[0]-1] # If single output
            
            # Handle Multiclass SHAP output format
            # shap_values could be a list (multiclass) or array (binary/regression)
            if isinstance(shap_values, list):
                # Multiclass: [n_samples, n_features] per class
                # We want the explanation for the 'High Risk' class for this single sample (index 0)
                feature_shap = shap_values[high_risk_idx][0]
            elif len(shap_values.shape) == 3:
                # Some versions return [n_samples, n_features, n_classes]
                 feature_shap = shap_values[0, :, high_risk_idx]
            else:
                # Binary/Single output: [n_samples, n_features]
                feature_shap = shap_values[0]

            # Get top features
            # Create list of (feature_name, shap_value)
            factors = []
            for name, val in zip(feature_order, feature_shap):
                factors.append({"feature": name, "value": float(val)})
            
            # Sort by absolute impact
            factors.sort(key=lambda x: abs(x['value']), reverse=True)
            
        else:
            # Fallback Logic
            print("Using fallback rule-based logic.")
            # Simple heuristic
            score = 0
            if input_data.Credit_Score < 600: score += 40
            if input_data.Past_Defaults > 0: score += 30
            ratio = input_data.Loan_Amount / (input_data.Annual_Income + 1)
            if ratio > 0.4: score += 20
            
            risk_probability = min(score / 100.0, 1.0)
            risk_score = risk_probability * 100
            
            if risk_score > 66: risk_level = "High"
            elif risk_score > 33: risk_level = "Medium"
            else: risk_level = "Low"
            
            # Mock SHAP
            factors = [
                {"feature": "Credit_Score", "value": -0.5 if input_data.Credit_Score > 700 else 0.5},
                {"feature": "Past_Defaults", "value": 1.0 if input_data.Past_Defaults > 0 else 0.0},
                {"feature": "Loan_Amount", "value": 0.3}
            ]

        # Financial Exposure = Loan Amount * Risk Probability
        exposure = input_data.Loan_Amount * risk_probability
        
        return {
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "risk_probability": float(risk_probability),
            "financial_exposure": float(exposure),
            "shap_factors": factors
        }

risk_model = RiskModel()

def get_prediction(input_data: PredictionInput):
    return risk_model.predict(input_data)
