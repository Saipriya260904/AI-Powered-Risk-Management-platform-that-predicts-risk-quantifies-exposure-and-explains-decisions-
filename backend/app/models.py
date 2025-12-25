from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="owner")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Inputs stored for reference
    input_data = Column(JSON)  # Stores {Age, Income, etc.}
    
    # Outputs
    risk_level = Column(String) # Low, Medium, High
    risk_score = Column(Float)  # 0-100
    risk_probability = Column(Float) # 0.0 - 1.0
    financial_exposure = Column(Float) # predicted loss amount
    
    # Explainability
    shap_factors = Column(JSON) # Top contributing features
    explanation = Column(Text)  # LLM generated text
    
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="predictions")
