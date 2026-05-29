@echo off
REM Single-robot visual demo training (verify smooth motion before full run)
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
echo ================================================
echo  G1 Inspire - DEMO Grasp-Lift (1 robot, GUI)
echo  Task: grasp red block + lift 5 cm
echo -----------------------------------------------
echo  Parallel envs : 1
echo  Train iters   : 100
echo  Checkpoints   : logs\g1_grasp_lift\checkpoints\
echo  Note: training uses random exploration; slight motion is normal.
echo ================================================
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%REPO_ROOT%\grasp_rl\train.py" --num_envs 1 --train_iters 100 --checkpoint_interval 50 --device cuda:0 %*
endlocal
