@echo off
REM Train G1 Inspire Hand grasp RL
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
echo ================================================
echo  G1 Inspire Hand - Grasp ^& Lift RL Training
echo  Task: grasp red block + lift 5 cm above table
echo -----------------------------------------------
echo  Device                : cuda:0 (GPU training)
echo  Default parallel envs : 16  (--num_envs N to override)
echo  Default train iters   : 2000 (--train_iters N to override)
echo  Default checkpoint    : every 1000 timesteps (--checkpoint_interval N)
echo  Checkpoints saved to  : logs\g1_grasp_lift_v7\checkpoints\
echo  Fine-tune from v6     : --resume "path\to\agent_79000.pt"
echo  Laptop tip            : use --headless --num_envs 4 if OOM
echo  Quick 1-robot demo    : run_train_demo.bat
echo ================================================
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%REPO_ROOT%\grasp_rl\train.py" --device cuda:0 %*
endlocal
