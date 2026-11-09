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

## Streamlit Hosting & Deployment

You can host and run the SmartBank BFSI AI Assistant directly via Streamlit:

```bash
streamlit run app.py
```
Or on Windows, double-click `run_streamlit.bat`.

### Deploying to Streamlit Community Cloud:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Select this repository, branch `main`, and set the main file path to `app.py`.
4. In **Settings > Secrets**, add your Gemini API Key:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
5. Click **Deploy**. Your assistant will be live and accessible to users worldwide with full RAG document analysis!

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
├── backend/
│   ├── main.py               # FastAPI entry point and API routes
│   ├── auth.py               # Registration, login and logout
│   ├── banking.py            # Account and transaction APIs
│   ├── chatbot.py             # Assistant and Gemini integration
│   ├── rag_service.py         # Local retrieval over banking guidance
│   ├── applications.py       # Loan application APIs
│   ├── portfolio.py          # Admin analytics APIs
│   ├── schemas.py             # Request validation
│   ├── security.py            # Password hashing and JWT sessions
│   ├── json_store.py          # JSON persistence
│   ├── data/                  # Demo JSON records
│   ├── models/                # Pre-trained ML pipelines
│   └── training/              # Reproducible model training and datasets
├── .env.example               # Safe configuration template
├── requirements.txt           # Python dependencies
└── README.md                  # Setup and usage guide
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
