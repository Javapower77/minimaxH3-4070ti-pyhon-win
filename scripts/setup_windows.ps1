[CmdletBinding()]
param(
    [switch]$SkipModelDownload,
    [switch]$SkipComfyInstall,
    [switch]$IncludeFullDiffusersBase
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-Python311 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidate = (& py -3.11 -c "import sys; print(sys.executable)").Trim()
        if ($LASTEXITCODE -eq 0 -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $version = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        if ($version -eq '3.11') {
            return (Get-Command python).Source
        }
    }

    throw 'Python 3.11 was not found. Install it or register it with the Windows py launcher.'
}

$Python311 = Resolve-Python311
$AppPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $AppPython)) {
    Write-Host 'Creating the application Python 3.11 environment...'
    & $Python311 -m venv (Join-Path $Root '.venv')
}

$ActualVersion = (& $AppPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($ActualVersion -ne '3.11') {
    throw ".venv uses Python $ActualVersion instead of 3.11. Delete .venv and rerun this script."
}

Write-Host 'Installing application dependencies...'
& $AppPython -m pip install --upgrade pip setuptools wheel
& $AppPython -m pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $AppPython -m pip install -r requirements.txt
& $AppPython -m pip install --no-deps -e .

if (-not $SkipModelDownload) {
    $backendArgs = @('scripts/setup_pruned_backend.py')
    if ($SkipComfyInstall) {
        $backendArgs += '--skip-install'
    }
    Write-Host 'Installing ComfyUI and downloading the low-memory model files...'
    & $AppPython @backendArgs
    Write-Host 'Downloading and validating the TaoMate LoRA...'
    & $AppPython -m minimax_h3_fl2v.download --loras --lora-id taomate_fl2va_3step_ema
    if ($IncludeFullDiffusersBase) {
        Write-Host 'Downloading the optional full Diffusers FL2VA snapshot...'
        & $AppPython -m minimax_h3_fl2v.download --base
    }
}

$RequiredFiles = @(
    '.runtime\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_fp8_scaled.safetensors',
    '.runtime\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    '.runtime\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors',
    '.runtime\ComfyUI\models\vae\minimax_h3_audio_vae_fp32.safetensors',
    '.runtime\ComfyUI\models\upscale_models\RealESRGAN_x4plus.safetensors',
    '.runtime\ComfyUI\models\frame_interpolation\rife_v4.25_lite.safetensors',
    '.runtime\ComfyUI\models\facerestore_models\codeformer.pth',
    '.runtime\ComfyUI\models\facedetection\detection_mobilenet0.25_Final.pth',
    '.runtime\ComfyUI\models\facedetection\parsing_parsenet.pth',
    '.runtime\ComfyUI\custom_nodes\facerestore_cf\__init__.py',
    '.runtime\ComfyUI\custom_nodes\minimax_h3_nodes\__init__.py',
    'models\loras\minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors'
)
$MissingFiles = @($RequiredFiles | Where-Object { -not (Test-Path (Join-Path $Root $_) -PathType Leaf) })
if ($MissingFiles.Count -gt 0) {
    throw "Low-memory backend setup is incomplete. Missing: $($MissingFiles -join ', ')"
}

Write-Host ''
Write-Host 'Windows setup complete.' -ForegroundColor Green
Write-Host 'The default pruned backend is ready; models\MiniMax-H3 is not required.'
Write-Host 'Start the studio with:'
Write-Host '  .\scripts\run_windows.ps1'
