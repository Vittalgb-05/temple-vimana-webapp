@echo off
cd /d "%~dp0"
echo Starting Temple Web App...
call .venv\Scripts\activate.bat
python src/app.py
pause
