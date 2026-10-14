@echo off
REM Re-runs only the RAG-v2 generation (retrieval ablation and the No-RAG-narrative arm are already done).
set PY=C:\Users\goura\anaconda3\envs\narrative-rag\python.exe
cd /d "%~dp0"
"%PY%" generate_v2.py
echo.
echo Finished. Tell Claude.
pause
