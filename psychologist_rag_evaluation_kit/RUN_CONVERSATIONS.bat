@echo off
REM Generates 13 constructs x (RAG, No-RAG) three-turn conversations with Gemma-2-9B. Needs Ollama + Qdrant (Docker) running.
docker start mh-qdrant >nul 2>&1
cd /d "%~dp0conversations"
"C:\Users\goura\anaconda3\envs\narrative-rag\python.exe" run_conversations.py --model-label Gemma-2-9B
echo.
echo Finished. Tell Claude.
pause
