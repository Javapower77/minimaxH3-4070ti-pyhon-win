[CmdletBinding()]
param(
    [int]$Port = 7868,
    [int]$ComfyPort = 18188
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    throw 'The Python environment is missing. Run .\scripts\setup_windows.ps1 first.'
}

$env:GRADIO_SERVER_PORT = [string]$Port
$env:MINIMAX_H3_COMFY_HOST = '127.0.0.1'
$env:MINIMAX_H3_COMFY_PORT = [string]$ComfyPort
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
Set-Location $Root
& $Python app.py
