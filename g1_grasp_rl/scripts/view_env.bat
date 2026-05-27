@echo off
REM View G1 grasp RL training scene
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  G1 Grasp RL - Scene Preview
echo ================================================
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%view_env.py" --device cpu
endlocal
