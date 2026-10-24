# 09-4 하루 실험 일정을 Windows 작업 스케줄러에서 실행하는 래퍼
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_daily.ps1 -Phase close
param([ValidateSet("open", "close")][string]$Phase)
Set-Location (Join-Path $PSScriptRoot "..")
$env:VIRTUAL_ENV = ""
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # 파이썬 출력(UTF-8)을 그대로 읽기
$log = "data\processed\paper_orders\scheduler_$(Get-Date -Format yyyyMMdd)_$Phase.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
"[$(Get-Date -Format s)] start $Phase" | Out-File -Append -Encoding utf8 $log
& $uv run python -m limitup.execution.daily $Phase 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
"[$(Get-Date -Format s)] end $Phase (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
