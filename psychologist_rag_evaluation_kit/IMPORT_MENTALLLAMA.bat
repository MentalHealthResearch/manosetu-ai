@echo off
setlocal EnableDelayedExpansion
REM Imports MentaLLaMA-chat-13B (TensorBlock GGUF) into Ollama as mental-llama:13b, then generates its 80 answers.
set KIT=%~dp0
set PY=C:\Users\goura\anaconda3\envs\narrative-rag\python.exe
set REPO=tensorblock/klyang_MentaLLaMA-chat-13B-GGUF
cd /d "%KIT%models"

REM --- Route 1: a .gguf you downloaded yourself into the models folder (must be > 1 GB) ---
set SRC=
for %%f in (*.gguf) do if %%~zf GTR 1000000000 set SRC=./%%f
if defined SRC (
  echo Using local file !SRC!
  goto CREATE
)

REM --- Route 2: let Ollama fetch it from Hugging Face (it resolves the file name itself) ---
for %%t in ("huggingface.co/%REPO%:Q2_K" "hf.co/%REPO%:Q2_K") do (
  if not defined SRC (
    echo.
    echo Trying: ollama pull %%~t
    ollama pull %%~t
    if !errorlevel! EQU 0 set SRC=%%~t
  )
)
if not defined SRC (
  echo.
  echo ==== Automatic download failed. ====
  echo Open https://huggingface.co/%REPO%/tree/main in your browser,
  echo download MentaLLaMA-chat-13B-Q2_K.gguf ^(4.85 GB^) into:
  echo   %KIT%models
  echo then double-click this file again.
  pause & exit /b 1
)

:CREATE
echo.
echo Creating mental-llama:13b from !SRC! with the LLaMA-2-chat template...
powershell -NoProfile -Command "(Get-Content Modelfile.mentalllama) -replace '\./MODEL\.gguf','!SRC!' | Set-Content Modelfile.local"
ollama create mental-llama:13b -f Modelfile.local
if errorlevel 1 (echo Import failed - send Claude a screenshot of this window. & pause & exit /b 1)
ollama run mental-llama:13b "Say hello in one sentence."
echo.
echo Generating the 80 MentalLLaMA answers ^(resumable^)...
cd /d "%KIT%benchmark"
"%PY%" run_benchmark.py --model MentalLLaMA-13B
echo.
echo Finished. Send Claude a message.
pause
