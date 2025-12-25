# Deployment Guide: AI Risk Management Platform

## Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL (Local or Cloud)
- OpenAI API Key (Optional, for LLM explanations)

## Backend Setup (FastAPI + ML)

1. **Navigate to backend:**
   ```bash
   cd backend
   ```

2. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train Model (First Run Only):**
   This generates the XGBoost model and artifacts.
   ```bash
   python ml/train_model.py
   ```

5. **Configure Environment:**
   Set environment variables (or create a `.env` file):
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost/risk_db"
   export OPENAI_API_KEY="your-api-key"
   ```

6. **Run Server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   API will be available at `http://localhost:8000`.
   Docs at `http://localhost:8000/docs`.

## Frontend Setup (React + Vite)

1. **Navigate to frontend:**
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies:**
   ```bash
   npm install
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   npm install recharts axios lucide-react
   ```
   *Note: Configuration files are already provided.*

3. **Run Development Server:**
   ```bash
   npm run dev
   ```
   Dashboard available at `http://localhost:5173`.

## Database Setup
- Ensure PostgreSQL is running.
- Create a database named `risk_db`.
- The application automatically creates tables on startup.

## Production Build
1. Build Frontend: `npm run build`
2. Serve static files via Nginx or FastAPI StaticFiles mount.
3. Run Backend with Gunicorn: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app`
