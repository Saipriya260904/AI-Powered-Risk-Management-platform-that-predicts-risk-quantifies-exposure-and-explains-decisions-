import os
from openai import OpenAI
from .schemas import PredictionInput

# Initialize Client
# Use a placeholder if env var is missing to avoid startup crash; actual call checks env var later
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing-key-placeholder")

def generate_explanation(input_data: PredictionInput, risk_data: dict):
    """
    Generates a plain English explanation for the risk prediction.
    """
    
    # Prepare text for LLM
    factors_text = ", ".join([f"{f['feature']} (Impact: {'High' if abs(f['value']) > 0.5 else 'Moderate'})" for f in risk_data['shap_factors'][:3]])
    
    prompt = f"""
    You are a helpful financial risk assistant. explain the following credit risk assessment in plain English to a non-technical loan officer.
    
    Applicant Profile:
    - Annual Income: ${input_data.Annual_Income}
    - Credit Score: {input_data.Credit_Score}
    - Loan Amount: ${input_data.Loan_Amount}
    - Past Defaults: {input_data.Past_Defaults}
    
    Model Prediction:
    - Risk Level: {risk_data['risk_level']}
    - Risk Probability: {risk_data['risk_probability']:.1%}
    - Key Factors Influencing Decision: {factors_text}
    
    Please provide a concise, 3-sentence explanation of why this risk level was assigned. Focus on the key drivers. Use a professional but easy-to-understand tone.
    """
    
    # Fallback if no API key (OpenAI client handles verify, but we can check the env var)
    if not os.getenv("OPENAI_API_KEY"):
        return f"The assessment indicates {risk_data['risk_level']} risk, primarily driven by {factors_text}. (AI explanation unavailable - API Key missing)"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial risk analyst expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Explanation could not be generated at this time."
