@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   GNN Conda Environment Setup Script
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM  STEP 1: Check if Conda is available
REM ---------------------------------------------------------------
echo [1/5] Checking for Conda...
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
echo   Conda found.
echo.

REM ---------------------------------------------------------------
REM  STEP 2: Check CUDA version via nvidia-smi
REM ---------------------------------------------------------------
echo [2/5] Checking for CUDA (nvidia-smi)...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   nvidia-smi not found. Checking if NVIDIA drivers need installation...
    echo.
    echo   No NVIDIA GPU driver detected on this machine.
    echo   You have two options:
    echo     A) Install NVIDIA drivers + CUDA manually from:
    echo        https://developer.nvidia.com/cuda-downloads
    echo     B) Continue with CPU-only PyTorch (no GPU acceleration)
    echo.
    set /p USER_CHOICE="Continue with CPU-only install? (y/n): "
    if /i "!USER_CHOICE!"=="y" (
        set CUDA_VERSION=cpu
        echo   Proceeding with CPU-only PyTorch.
    ) else (
        echo   Exiting. Please install CUDA and re-run this script.
        pause
        exit /b 1
    )
) else (
    REM Parse CUDA version from nvidia-smi output
    for /f "tokens=*" %%i in ('nvidia-smi ^| findstr /i "CUDA Version"') do (
        set CUDA_LINE=%%i
    )
    echo   Raw nvidia-smi CUDA line: !CUDA_LINE!

    REM Extract major version number (e.g., 11, 12, 13)
    for /f "tokens=3" %%v in ("!CUDA_LINE!") do set CUDA_FULL=%%v
    for /f "delims=." %%m in ("!CUDA_FULL!") do set CUDA_MAJOR=%%m

    echo   Detected CUDA version: !CUDA_FULL! (Major: !CUDA_MAJOR!)

    REM Map CUDA major version to PyTorch wheel suffix
    if "!CUDA_MAJOR!"=="13" set CUDA_VERSION=cu130
    if "!CUDA_MAJOR!"=="12" set CUDA_VERSION=cu121
    if "!CUDA_MAJOR!"=="11" set CUDA_VERSION=cu118
    if "!CUDA_MAJOR!"=="10" set CUDA_VERSION=cu102

    if not defined CUDA_VERSION (
        echo.
        echo   WARNING: CUDA major version !CUDA_MAJOR! is not mapped.
        echo   Defaulting to cu121. Edit CUDA_VERSION in this script if wrong.
        set CUDA_VERSION=cu121
    )

    echo   Using PyTorch wheel index: !CUDA_VERSION!
)
echo.

REM ---------------------------------------------------------------
REM  STEP 3: Create Conda environment
REM ---------------------------------------------------------------
echo [3/5] Creating conda environment 'gnn' with Python 3.12...
call conda create -n gnn python=3.12 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)
echo   Environment created.
echo.

REM ---------------------------------------------------------------
REM  STEP 4: Activate and install PyTorch
REM ---------------------------------------------------------------
echo [4/5] Installing PyTorch for !CUDA_VERSION!...

if "!CUDA_VERSION!"=="cpu" (
    call conda run -n gnn pip install torch torchvision torchaudio
) else (
    call conda run -n gnn pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CUDA_VERSION!
)

if errorlevel 1 (
    echo ERROR: PyTorch installation failed.
    pause
    exit /b 1
)
echo   PyTorch installed.
echo.

REM ---------------------------------------------------------------
REM  STEP 5: Install remaining packages
REM ---------------------------------------------------------------
echo [5/5] Installing torch_geometric, ogb, and other packages...
call conda run -n gnn pip install torch_geometric
call conda run -n gnn pip install ogb networkx tqdm seaborn matplotlib jupyter notebook

if errorlevel 1 (
    echo ERROR: One or more packages failed to install.
    pause
    exit /b 1
)
echo   All packages installed.
echo.

echo ============================================================
echo   Setup Complete!
echo   To activate your environment, run:
echo     conda activate gnn
echo ============================================================
echo.
pause