from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class PredictionInput(BaseModel):
    Age: int
    Annual_Income: float
    Loan_Amount: float
    Credit_Score: int
    Employment_Years: int
    Past_Defaults: int

class PredictionOutput(BaseModel):
    id: int
    risk_level: str
    risk_score: float
    risk_probability: float
    financial_exposure: float
    shap_factors: List[Dict[str, Any]] # e.g. [{"feature": "Age", "value": 0.2}]
    explanation: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
