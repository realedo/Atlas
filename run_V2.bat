@echo off
title Atlas Server - ExLlamaV2 
echo Starting up ExLlamaV2 server...
cd /d "%~dp0"
call venv\Scripts\activate.bat
cd backend

python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause