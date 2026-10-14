@echo off
REM RAG-v2: runs alongside the existing pipeline; nothing in mental_health_research\ or the kit is modified.
REM First run downloads BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 (~4.5 GB) into mental_health_research\.hf_cache
REM Needs: Ollama running with gemma2:9b. Docker/Qdrant NOT needed (exact in-memory search over 735 episodes).
set PY=C:\Users\goura\anaconda3\envs\narrative-rag\python.exe
cd /d "%~dp0"
echo ===== 1/3 Retrieval ablation A0-A5 =====
"%PY%" eval_retrieval.py || goto :err
echo ===== 2/3 Full RAG-v2 generation (40 questions) =====
"%PY%" generate_v2.py || goto :err
echo ===== 3/3 Control arm: No-RAG with narratives from model memory =====
"%PY%" generate_v2.py --parametric || goto :err
echo.
echo Finished. Results are in rag_v2\outputs. Tell Claude.
pause
exit /b 0
:err
echo.
echo Something failed - send Claude the last lines above.
pause
