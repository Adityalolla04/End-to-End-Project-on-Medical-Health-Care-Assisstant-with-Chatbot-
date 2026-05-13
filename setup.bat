@echo off
:: ══════════════════════════════════════════════════════════════════
::  Medical Healthcare Chatbot — One-Click Setup (Windows)
::  Compatible with Python 3.10, 3.11, 3.12
::  Run from VS Code terminal:  .\setup.bat
:: ══════════════════════════════════════════════════════════════════
setlocal EnableDelayedExpansion
title Medical Chatbot Setup
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║   Medical Healthcare AI Chatbot — Setup          ║
echo  ║   RAG + FastAPI + Streamlit                      ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

:: ── Check Python version ──────────────────────────────────────────
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause & exit /b 1
)

:: ── STEP 1: Create virtual environment ────────────────────────────
echo [1/8] Creating Python virtual environment...
if exist venv\Scripts\activate.bat (
    echo   venv already exists — skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create venv.
        pause & exit /b 1
    )
    echo   Done.
)

:: ── STEP 2: Activate venv ─────────────────────────────────────────
echo [2/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv.
    pause & exit /b 1
)
echo   Done. Python: %VIRTUAL_ENV%

:: ── STEP 3: Upgrade pip ───────────────────────────────────────────
echo [3/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo   Done.

:: ── STEP 4: Install core web + config packages ────────────────────
echo [4/8] Installing core packages (fastapi, langchain, chromadb)...
pip install --quiet ^
    "fastapi>=0.111.0" ^
    "uvicorn[standard]>=0.29.0" ^
    "python-multipart>=0.0.9" ^
    "websockets>=12.0" ^
    "langchain>=0.2.0" ^
    "langchain-community>=0.2.0" ^
    "langchain-core>=0.2.0" ^
    "chromadb>=0.5.0" ^
    "python-dotenv>=1.0.1" ^
    "pydantic>=2.7.0" ^
    "pydantic-settings>=2.2.0"
echo   Done.

:: ── STEP 5: Install data + ML packages ────────────────────────────
echo [5/8] Installing data and ML packages...
pip install --quiet ^
    "pandas>=2.2.0" ^
    "numpy>=1.26.0" ^
    "scikit-learn>=1.5.0" ^
    "joblib>=1.4.0" ^
    "plotly>=5.22.0"
echo   Done.

:: ── STEP 6: Install sentence-transformers (embeddings) ────────────
echo [6/8] Installing sentence-transformers for RAG embeddings...
echo   This downloads ~500MB — please wait...
:: Kill any lingering hf.exe processes to avoid WinError 32
taskkill /f /im hf.exe >nul 2>&1
pip install --quiet "sentence-transformers>=3.0.0"
if errorlevel 1 (
    echo   WARNING: sentence-transformers install had issues.
    echo   Try running manually: pip install sentence-transformers
    echo   Continuing setup...
) else (
    echo   Done.
)

:: ── STEP 7: Install spaCy (Python 3.12 needs 3.8+) ───────────────
echo [7/8] Installing spaCy and downloading language model...
pip install --quiet "spacy>=3.8.0" "streamlit>=1.35.0" pytest httpx
if errorlevel 1 (
    echo   WARNING: spaCy install had issues.
)
:: Download spaCy model
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo   WARNING: spaCy model download failed (needs internet).
    echo   Run manually later: python -m spacy download en_core_web_sm
)
echo   Done.

:: ── STEP 8: Create .env and run tests ────────────────────────────
echo [8/8] Final setup and verification...
if not exist .env (
    copy .env.example .env >nul
    echo   Created .env from template.
    echo   OPEN .env and set LLM_PROVIDER=ollama (or claude/openai)
) else (
    echo   .env already exists.
)

:: Run the tests
echo.
echo   Running 10 API tests...
python -m pytest tests\test_api.py -v -p no:cacheprovider --tb=short
if errorlevel 1 (
    echo   Some tests failed — check output above.
) else (
    echo   All 10 tests PASSED!
)

:: ── Summary ────────────────────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║   Setup Complete!                                ║
echo  ║                                                  ║
echo  ║   NEXT STEPS (run in order):                     ║
echo  ║                                                  ║
echo  ║   1. Edit .env — set LLM_PROVIDER                ║
echo  ║      Options: ollama, claude, openai, local      ║
echo  ║                                                  ║
echo  ║   2. Build RAG vector store (one-time, ~3 min):  ║
echo  ║      python -m rag.ingest                        ║
echo  ║                                                  ║
echo  ║   3. Start the app:                              ║
echo  ║      .\start_app.bat                             ║
echo  ║                                                  ║
echo  ║   URLs after start:                              ║
echo  ║      Chatbot UI  → http://localhost:8501         ║
echo  ║      API Docs    → http://localhost:8000/docs    ║
echo  ╚═══════════════════════════════════════════════════╝
echo.
pause
