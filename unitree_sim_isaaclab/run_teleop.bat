@echo off
REM Keyboard teleop: G1 right arm + red cylinder (same scene as run_train.bat)
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
set "CONDA_ROOT=E:\Miniconda3"
set "CONDA_ENV=env_isaaclab"

echo ================================================
echo  Red Block Grasp - Pink IK Teleop (right arm only)
echo  Requires: conda env_isaaclab + pinocchio
echo -----------------------------------------------
echo  Move: W/S A/D Q/E   Rotate: Z/X T/G C/V
echo  RIGHT ARM ONLY - left arm locked at reset pose
echo  Hand: K close / N open   Reset: R
echo  Tip: stop training Isaac Sim before starting (GPU lock)
echo ================================================

if exist "%CONDA_ROOT%\Scripts\activate.bat" (
    call "%CONDA_ROOT%\Scripts\activate.bat" %CONDA_ENV%
) else (
    echo [WARN] Conda not found at %CONDA_ROOT%. Activate %CONDA_ENV% manually if startup fails.
)

if not defined CONDA_PREFIX (
    echo [ERROR] Conda env "%CONDA_ENV%" is not active. Run: conda activate %CONDA_ENV%
    exit /b 1
)

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%teleop_grasp.py" --device cuda:0 %*
endlocal
