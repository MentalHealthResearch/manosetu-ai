@echo off
REM Qwen3-8B through the same 40-question RAG / No-RAG benchmark as the other four models.
set PY=C:\Users\goura\anaconda3\envs\narrative-rag\python.exe

echo [1/4] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
  echo Starting Docker Desktop - this can take 1-2 minutes...
  start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
  if not exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
)
set /a n=0
:waitdocker
docker info >nul 2>&1
if not errorlevel 1 goto dockerok
set /a n+=1
if %n% GEQ 60 (echo Docker did not start. Open Docker Desktop manually, then run this file again. & pause & exit /b 1)
timeout /t 5 >nul
goto waitdocker
:dockerok

echo [2/4] Starting Qdrant...
docker start mh-qdrant >nul 2>&1
if errorlevel 1 docker run -d --name mh-qdrant -p 6333:6333 -v mh_qdrant_storage:/qdrant/storage qdrant/qdrant >nul
set /a n=0
:waitq
powershell -NoProfile -Command "try{Invoke-RestMethod http://localhost:6333/collections -TimeoutSec 3 | Out-Null; exit 0}catch{exit 1}"
if not errorlevel 1 goto qok
set /a n+=1
if %n% GEQ 30 (echo Qdrant did not respond on port 6333. & pause & exit /b 1)
timeout /t 3 >nul
goto waitq
:qok
powershell -NoProfile -Command "$c=(Invoke-RestMethod http://localhost:6333/collections).result.collections.name; if($c -notcontains 'narrative_therapy_corpus'){exit 1}"
if errorlevel 1 (
  echo Collection missing - rebuilding the narrative_therapy_corpus index...
  pushd "%~dp0..\mental_health_research"
  "%PY%" scripts\ingest_all_narrative_corpus.py --dataset-root data\dataset
  popd
)

echo [3/4] Pulling qwen3:8b...
ollama pull qwen3:8b

echo [4/4] Generating Qwen3 responses (resumable)...
cd /d "%~dp0benchmark"
"%PY%" run_qwen3.py
echo.
echo Finished. Tell Claude.
pause
