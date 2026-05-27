@echo off
REM G1 Inspire Hand grasp-and-lift RL training (SKRL PPO)
REM Usage: run_train.bat [--num_envs 128] [--train_iters 2000]

setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"

echo ================================================
echo  G1 Inspire Hand - Grasp and Lift RL Training
echo ================================================

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p "%SCRIPT_DIR%train.py" %*
endlocal
