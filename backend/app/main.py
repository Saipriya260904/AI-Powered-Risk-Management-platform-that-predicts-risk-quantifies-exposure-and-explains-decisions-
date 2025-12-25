from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from . import models, schemas, database, ml_service, llm_service

# Initialize DB tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI Risk Management Platform", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    return database.get_db()

@app.get("/")
def read_root():
    return {"message": "Risk Management API is running"}

@app.post("/predict", response_model=schemas.PredictionOutput)
def predict_risk(input_data: schemas.PredictionInput, db: Session = Depends(database.get_db)):
    """
    End-to-end Risk Prediction Pipeline:
    1. Receive Applicant Data
    2. ML Model Prediction (XGBoost)
    3. Exposure Calculation
    4. SHAP Explanation
    5. LLM Narrative Generation
    6. Store in DB
    """
    
    # 1. Get ML Prediction & SHAP
    risk_data = ml_service.get_prediction(input_data)
    
    # 2. Get LLM Explanation
    explanation = llm_service.generate_explanation(input_data, risk_data)
    
    # 3. Create Record
    # We are using a dummy user_id=1 for this demo since Auth is mocked for the predicting part usually
    # In a real app, we'd get current_user from token
    
    # Let's check if user 1 exists, create if not (demo convenience)
    user = db.query(models.User).filter(models.User.id == 1).first()
    if not user:
        user = models.User(username="demo_officer", hashed_password="hashed_secret")
        db.add(user)
        db.commit()
        db.refresh(user)

    db_prediction = models.Prediction(
        user_id=user.id,
        input_data=input_data.model_dump(),
        risk_level=risk_data["risk_level"],
        risk_score=risk_data["risk_score"],
        risk_probability=risk_data["risk_probability"],
        financial_exposure=risk_data["financial_exposure"],
        shap_factors=risk_data["shap_factors"],
        explanation=explanation
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return db_prediction

@app.get("/history", response_model=List[schemas.PredictionOutput])
def get_history(skip: int = 0, limit: int = 20, db: Session = Depends(database.get_db)):
    predictions = db.query(models.Prediction).order_by(models.Prediction.created_at.desc()).offset(skip).limit(limit).all()
    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
