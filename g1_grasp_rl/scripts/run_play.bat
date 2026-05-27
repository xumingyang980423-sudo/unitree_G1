@echo off
REM G1 Inspire Hand grasp-and-lift RL play (load trained model)
REM Usage: run_play.bat [--checkpoint path/to/checkpoint.pt]

setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"

echo ================================================
echo  G1 Inspire Hand - Grasp and Lift RL Play
echo ================================================

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p "%SCRIPT_DIR%play.py" --device cpu %*
endlocal
