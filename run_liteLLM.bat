@echo off
title Atlas litellm Server 
echo Starting up litellm...
cd /d "%~dp0"
cd backend_exv3
call venv_v3\Scripts\activate.bat


litellm --config litellm_config.yaml --port 4000 --host 0.0.0.0
pause