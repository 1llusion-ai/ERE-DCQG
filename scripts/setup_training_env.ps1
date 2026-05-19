param(
    [string]$VenvPath = ".venv-deberta",
    [string]$Python = "python",
    [ValidateSet("cpu", "cu121", "skip")]
    [string]$Torch = "cpu"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment: $VenvPath"
if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $Python -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Could not find venv python at $venvPython"
}

Write-Host "Upgrading pip tooling"
& $venvPython -m pip install --upgrade pip setuptools wheel

if ($Torch -eq "cpu") {
    Write-Host "Installing CPU PyTorch"
    & $venvPython -m pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
} elseif ($Torch -eq "cu121") {
    Write-Host "Installing CUDA 12.1 PyTorch"
    & $venvPython -m pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Skipping PyTorch install"
}

Write-Host "Installing training dependencies"
& $venvPython -m pip install -r requirements-train.txt

Write-Host "Verifying imports"
@'
import numpy
import sklearn
import torch
import transformers
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
'@ | & $venvPython -

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Run smoke test with:"
Write-Host "  .\$VenvPath\Scripts\python.exe -m scripts.train_classifier --smoke_test --device cpu --output_dir outputs/models/smoke_deberta_multitask"
