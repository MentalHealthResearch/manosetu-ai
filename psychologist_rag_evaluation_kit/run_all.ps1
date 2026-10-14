# One-click benchmark runner: Ollama models + Qdrant + ingest + 320 responses.
# Right-click > "Run with PowerShell" (or: powershell -ExecutionPolicy Bypass -File run_all.ps1)
$ErrorActionPreference = "Continue"
$Kit      = $PSScriptRoot
$Project  = Join-Path (Split-Path $Kit -Parent) "mental_health_research"
$Bench    = Join-Path $Kit "benchmark"
$Log      = Join-Path $Bench "run_log.txt"
$Py       = "C:\Users\goura\anaconda3\envs\narrative-rag\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:HF_HOME = Join-Path $Project ".hf_cache"
$env:PYTHONIOENCODING = "utf-8"
Start-Transcript -Path $Log -Append | Out-Null
function Step($m) { Write-Host "`n==== [$(Get-Date -Format HH:mm:ss)] $m ====" -ForegroundColor Cyan }

Step "1/6 Python packages"
& $Py -m pip install -q ollama qdrant-client sentence-transformers pandas

Step "2/6 Ollama server"
try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 5 | Out-Null }
catch {
  Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe" -ErrorAction SilentlyContinue
  Start-Sleep 15
}

Step "3/6 Pull models"
foreach ($t in "llama3.1:8b","mistral:7b","gemma2:9b") { ollama pull $t }
# MentalLLaMA is not in the official Ollama library; try GGUF builds of klyang/MentaLLaMA-chat-13B.
$mentalTag = $null
ollama show mental-llama:13b *> $null
if ($LASTEXITCODE -eq 0) { $mentalTag = "mental-llama:13b" }
if (-not $mentalTag) {
foreach ($t in "huggingface.co/mradermacher/MentaLLaMA-chat-13B-GGUF:Q4_K_M",
               "huggingface.co/mradermacher/MentaLLaMA-chat-13B-i1-GGUF:Q4_K_M",
               "huggingface.co/QuantFactory/MentaLLaMA-chat-13B-GGUF:Q4_K_M",
               "huggingface.co/TheBloke/MentaLLaMA-chat-13B-GGUF:Q4_K_M",
               "huggingface.co/mradermacher/MentaLLaMA-chat-13B-GGUF") {
  ollama pull $t
  if ($LASTEXITCODE -eq 0) { $mentalTag = $t; break }
}
}
if ($mentalTag) {
  Write-Host "MentalLLaMA tag: $mentalTag"
  $cfgPath = Join-Path $Bench "benchmark_config.json"
  $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
  $cfg.models."MentalLLaMA-13B" = $mentalTag
  $cfg | ConvertTo-Json -Depth 5 | Set-Content $cfgPath -Encoding UTF8
} else { Write-Host "WARNING: no MentalLLaMA GGUF could be pulled; that model will be skipped." -ForegroundColor Yellow }

Step "4/6 Qdrant (Docker)"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
  Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe" -ErrorAction SilentlyContinue
  if (-not $?) { Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -ErrorAction SilentlyContinue }
  for ($i=0; $i -lt 30; $i++) { Start-Sleep 5; docker info *> $null; if ($LASTEXITCODE -eq 0) { break } }
}
$exists = docker ps -a --filter "name=^mh-qdrant$" --format "{{.Names}}"
if ($exists) { docker start mh-qdrant | Out-Null }
else { docker run -d --name mh-qdrant -p 6333:6333 -v mh_qdrant_storage:/qdrant/storage qdrant/qdrant | Out-Null }
for ($i=0; $i -lt 20; $i++) { try { Invoke-RestMethod http://localhost:6333/collections -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep 3 } }

Step "5/6 Ingest narrative_therapy_corpus (skipped if present)"
$cols = (Invoke-RestMethod http://localhost:6333/collections).result.collections.name
if ($cols -notcontains "narrative_therapy_corpus") {
  Push-Location $Project
  & $Py scripts\ingest_all_narrative_corpus.py --dataset-root data\dataset
  Pop-Location
} else { Write-Host "Collection already exists." }

Step "6/6 Generate 320 responses (resumable - rerun this script if interrupted)"
Push-Location $Bench
if ($mentalTag) { & $Py run_benchmark.py }
else { & $Py run_benchmark.py --model Llama-3.1-8B --model Mistral-7B --model Gemma-2-9B }
Pop-Location

Step "DONE"
Get-ChildItem (Join-Path $Kit "generated_responses") | Format-Table Name, Length
Stop-Transcript | Out-Null
Read-Host "Finished. Press Enter to close"
