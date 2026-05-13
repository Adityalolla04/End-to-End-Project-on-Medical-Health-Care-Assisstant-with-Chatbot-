@echo off
echo ============================================
echo   MedBI -- Medical Intelligence Platform
echo ============================================

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start FastAPI backend (watches only source dirs, NOT venv)
echo.
echo [1/2] Starting FastAPI backend on http://localhost:8000
echo       Dashboard: http://localhost:8000
echo       API Docs:  http://localhost:8000/docs
echo.
start "MedBI API" cmd /k "venv\Scripts\activate.bat && uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir deployment --reload-dir rag"

:: Wait for API to start
timeout /t 3 /nobreak >nul

:: Open browser
echo [2/2] Opening dashboard in browser...
start http://localhost:8000

echo.
echo ============================================
echo   App is running!
echo   Press Ctrl+C in each window to stop.
echo ============================================
