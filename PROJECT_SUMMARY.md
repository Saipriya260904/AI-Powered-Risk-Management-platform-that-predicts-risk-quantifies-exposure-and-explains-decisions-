# AI-Powered Risk Management Platform
**Role:** Full Stack AI Engineer

## Project Description
Designed and built an end-to-end Risk Management Platform for the credit domain, utilizing Machine Learning to predict loan default risks and Generative AI to provide explainable insights. The system quantifies financial exposure and presents complex ML inference results through a user-friendly dashboard.

## Technical Stack
- **ML/AI:** XGBoost, SHAP (Explainable AI), OpenAI GPT-3.5, Scikit-learn, Pandas.
- **Backend:** FastAPI (Python), SQLAlchemy, Pydantic, PostgreSQL.
- **Frontend:** React.js, Tailwind CSS, Recharts (Data Visualization).
- **DevOps:** RESTful APIs, Git, Virtual Environments.

## Key Features & Achievements
- **Predictive Modeling:** Developed an XGBoost classifier achieving high accuracy in predicting determining 'Low', 'Medium', and 'High' risk categories based on applicant financial history.
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) to decompose model predictions, identifying key drivers (e.g., Credit Score, DTI ratio) for each decision.
- **Generative AI Integration:** Implemented an LLM pipeline to translate numerical SHAP values into plain English narratives, enabling non-technical stakeholders to understand risk factors.
- **Financial Exposure Engine:** Built a calculation engine to quantify potential financial loss based on loan amount and predicted default probability.
- **Interactive Dashboard:** Created a modern, responsive React UI featuring real-time risk gauges, interactive factor charts, and instant assessment feedback.
- **Scalable Architecture:** Designed a decoupled architecture with a FastAPI backend serving a React frontend, ensuring separation of concerns and maintainability.
