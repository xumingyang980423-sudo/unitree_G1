@echo off
setlocal EnableExtensions
REM Headless policy eval (low VRAM, no GUI)

set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "CKPT=%REPO_ROOT%\logs\g1_grasp_lift_v7\checkpoints\agent_1000.pt"

if not "%~1"=="" set "CKPT=%~1"

echo ================================================
echo  G1 Grasp-Lift Policy Eval (headless)
echo  Checkpoint: %CKPT%
echo  Tip: close other apps if OOM; reboot after training
echo ================================================

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call "%ISAACLAB_DIR%\isaaclab.bat" -p "%REPO_ROOT%\grasp_rl\eval_policy.py" --device cuda:0 --headless --checkpoint "%CKPT%" --episodes 10 --max_steps 200 %*
endlocal
