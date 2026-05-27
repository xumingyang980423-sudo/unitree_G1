@echo off
REM Train Dexsuite Kuka+Allegro dexterous lift
setlocal
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  Dexsuite Kuka+Allegro - Dexterous Lift RL
echo  Task: Isaac-Dexsuite-Kuka-Allegro-Lift-v0
echo ================================================
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%train_dexsuite.py" %*
endlocal
