@echo off
title Atlas Server - ExLlamaV3 
echo Starting up ExLlamaV3 server...
cd /d "%~dp0"
cd backend_exv3
call venv_v3\Scripts\activate.bat


python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause