@echo off
title Drift2Act - Clinical Drift Monitor
echo.
echo  ========================================
echo   Drift2Act v1.0 - Starting Dashboard
echo  ========================================
echo.

cd /d "%~dp0"

echo  Starting Streamlit server...
echo  Dashboard will open at http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.

python -m streamlit run dashboard/app.py

pause
