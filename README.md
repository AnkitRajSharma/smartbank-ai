# SmartBank AI

A local banking demo built with **FastAPI + plain HTML/CSS/JavaScript**. It uses fictional customer data stored in JSON files. It includes sign-in/sign-out, account dashboard, transactions, credit profile, loan applications and estimates, fraud estimates, portfolio analytics, and a banking assistant with optional Gemini and local FAQ/RAG fallback.

> **Demo only:** all accounts, transactions, credit profiles, and model datasets are synthetic. This is not a real banking system. It cannot move money, approve loans, or detect real-world fraud. Never use the sample credentials or demo data in production.

## 1. Requirements

- Python 3.11, 3.12, or 3.13
- Internet connection for the first dependency installation
- Optional: a Gemini API key for Gemini responses. The app still runs without it.

## 2. Install and run

Open a terminal in the `smartbank-ai` project folder.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn backend.main:app --reload
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000/app/login.html**. API documentation: **http://127.0.0.1:8000/docs**. Keep the terminal running while using the app.

For subsequent runs, activate `.venv` and run only:

```bash
uvicorn backend.main:app --reload
```

## Cloud Deployment (Render, Railway, etc.)

This repository is pre-configured for one-click deployment on [Render](https://render.com) using the included `render.yaml` Blueprint:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Environment Variables:**
  - `PYTHON_VERSION`: `3.11.9`
  - `GEMINI_API_KEY`: *(Optional)* Your Gemini API key for the AI assistant.

> Root requests (`/`) automatically redirect to `/app/login.html`, and `uvicorn main:app` can also be used interchangeably from the root folder.

## 3. Demo login

| Role | Email | Password |
|---|---|---|
| Customer 1 | `customer1@smartbank.demo` | `Aspen!Orbit57` |
| Customer 2 | `customer2@smartbank.demo` | `Harbor!Cedar84` |
| Customer 3 | `customer3@smartbank.demo` | `Mosaic!Falcon29` |
| Analytics admin | `analytics@smartbank.demo` | `Prism!Summit63` |

These credentials are for this local demo only. If the login data is missing or has been changed, run the seeder from the project root:

```bash
python backend/training/seed_banking_data.py
```

## 4. Optional Gemini setup

Install the optional SDK with `pip install -r requirements-gemini.txt`, then open `.env` and add your key to `GEMINI_API_KEY`. Leave it blank to use the built-in local assistant. Restart the backend after changing `.env`. Never commit or share `.env`.

## 5. Project structure

```text
smartbank-ai/
├── frontend/                 # HTML pages, CSS and browser JavaScript
│   ├── index.html            # Unified BFSI Banking Assistant UI
│   ├── dashboard.html        # Customer banking overview & cards
│   ├── style.css             # Cohesive styling & design system
│   └── script.js             # Client API integration & RAG upload
├── backend/
│   ├── main.py               # FastAPI entry point and API routes
│   ├── auth.py               # Registration, login and logout
│   ├── banking.py            # Account and transaction APIs
│   ├── chatbot.py            # Senior BFSI Advisory Specialist & Gemini LLM
│   ├── rag_service.py        # Local & user document RAG retrieval engine
│   ├── applications.py       # Loan application APIs
│   ├── portfolio.py          # Admin analytics APIs
│   ├── schemas.py            # Request validation
│   ├── security.py           # Password hashing and JWT sessions
│   ├── json_store.py         # JSON persistence
│   ├── data/
│   │   ├── knowledge_base/   # Comprehensive BFSI Advisory Markdown documents
│   │   └── user_documents/   # Isolated per-user uploaded documents (RAG indexed)
│   ├── models/               # Pre-trained ML pipelines
│   └── training/             # Reproducible model training and datasets
├── main.py                   # Root entrypoint alias for cloud environments
├── render.yaml               # Render Infrastructure as Code blueprint
├── .env.example              # Safe configuration template
├── requirements.txt          # Python dependencies (FastAPI + RAG + Gemini)
└── README.md                 # Setup and usage guide
```

## 6. Model training (optional)

Pre-trained model files are included. You do **not** need to train them to start the app. To regenerate them, activate the virtual environment and run:

```bash
python backend/training/train_loan.py
python backend/training/train_fraud.py
```

Training creates synthetic datasets and model reports under `backend/training/` and saves pipelines under `backend/models/`.

## 7. Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv`, then run `pip install -r requirements.txt` again.
- **Port 8000 is busy:** run `uvicorn backend.main:app --reload --port 8001` and open `http://127.0.0.1:8001/app/login.html`.
- **Model unavailable:** confirm both `.pkl` files exist in `backend/models/`. Retrain using the commands above if necessary.
- **Login fails:** run the seeder command in section 3, then restart the backend.
- **Gemini unavailable:** check `GEMINI_API_KEY` and network access. The local assistant fallback remains available.

## 8. Safety and limitations

Passwords for registered users are hashed. The demo login credentials are provided only for local testing. The JSON store is intended for a single-process demonstration, not production or multi-worker deployment. The included ML models are experimental and trained on synthetic data; their scores do not establish real lending or fraud-detection performance.
