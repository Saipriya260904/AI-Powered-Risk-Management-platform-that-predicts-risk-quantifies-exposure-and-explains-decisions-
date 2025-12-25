import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import shap
import os

# 1. Dataset Selection and Preprocessing (Synthetic Data Generation)
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    
    # Features
    age = np.random.randint(21, 70, n_samples)
    annual_income = np.random.randint(20000, 200000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    past_defaults = np.random.randint(0, 5, n_samples)
    
    # Loan amount correlated with income but with noise
    loan_amount = (annual_income * np.random.uniform(0.1, 0.6, n_samples)).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': annual_income,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Employment_Years': employment_years,
        'Past_Defaults': past_defaults
    })
    
    # Define Risk Logic (Ground Truth for Training)
    # We want a realistic distribution.
    # Risk Score calculation (strictly for labeling purposes)
    # Lower score = Higher Risk in this formula, but we'll map it to classes
    
    base_score = 0
    # Higher credit score reduces risk
    base_score += (credit_score - 300) / 550 * 50 
    # Valid income-to-loan ratio reduces risk
    dti = loan_amount / (annual_income + 1)
    base_score -= dti * 20
    # Past defaults heavily increase risk (reduce score)
    base_score -= past_defaults * 15
    # Employment stability helps
    base_score += employment_years * 1
    
    # Add noise
    base_score += np.random.normal(0, 5, n_samples)
    
    # Determine Labels
    # We explicitly want "Low", "Medium", "High"
    # Let's define thresholds based on percentiles to ensure balanced classes
    
    threshold_low = np.percentile(base_score, 33)
    threshold_high = np.percentile(base_score, 66)
    
    conditions = [
        (base_score < threshold_low), # High Risk (Low score)
        (base_score >= threshold_low) & (base_score < threshold_high), # Medium
        (base_score >= threshold_high) # Low Risk (High score)
    ]
    choices = ['High', 'Medium', 'Low']
    
    df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
    
    return df

def train():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # 2. Feature Engineering
    X = df.drop(columns=['Risk_Level'])
    y = df['Risk_Level']
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Note mapping: Check le.classes_ to ensure correct ordering interpretations later
    # We expected 'High', 'Low', 'Medium'. Alphabetical order: High=0, Low=1, Medium=2
    # This might be confusing. Let's force a specific mapping if needed, 
    # but for now we'll just save the encoder.
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scaling (Optional for XGBoost but good only for specific linear interpretations, 
    # keeping it raw is often fine for trees, but let's scale to standardize inputs for the API)
    # Actually, let's KEEP it unscaled for SHAP explainability to be easier (values in original units)
    # XGBoost handles unscaled data well.
    
    # 3. Model Training (XGBoost)
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    # 4. Save Artifacts
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
        
    joblib.dump(model, 'artifacts/risk_model.joblib')
    joblib.dump(le, 'artifacts/label_encoder.joblib')
    # We'll save a small sample of X_train for SHAP background distribution if needed
    # But usually TreeExplainer works fine without it or with just the model.
    # However, for API speed we might initialize the explainer once.
    
    print("Model and artifacts saved to backend/ml/artifacts/")

if __name__ == "__main__":
    train()
