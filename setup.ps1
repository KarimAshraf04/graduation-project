# ============================================================
#  Vision project - one-click setup (Windows, NVIDIA GPU)
#  Run from C:\vision :   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# ============================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# 1) Python check
Step "Checking Python"
try { $pyver = & python --version 2>&1 } catch { $pyver = "" }
if ($pyver -notmatch "Python 3\.12") {
    Write-Host "Python 3.12 not found (got: '$pyver'). Installing with winget..." -ForegroundColor Yellow
    winget install -e --id Python.Python.3.12 --override "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
python --version

# 2) Virtual environment
Step "Creating virtual environment (venv)"
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    python -m venv venv
} else {
    Write-Host "venv already exists - reusing it."
}
$py = ".\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

# 3) PyTorch with CUDA
Step "Installing PyTorch (CUDA 12.8 build)"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4) Project libraries
Step "Installing project libraries"
& $py -m pip install -r requirements.txt

# 5) onnxruntime GPU (clean install, with its own CUDA/cuDNN DLLs)
Step "Installing onnxruntime-gpu (for face models)"
& $py -m pip uninstall -y onnxruntime onnxruntime-gpu 2>$null
& $py -m pip install "onnxruntime-gpu[cuda,cudnn]"

# 6) Verify
Step "Verifying GPU"
& $py check_gpu.py

Write-Host "`nSetup finished. Start the app with:  .\run.bat" -ForegroundColor Green
