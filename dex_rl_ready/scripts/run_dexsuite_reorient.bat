@echo off
REM Train Dexsuite Kuka+Allegro dexterous reorient
setlocal
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  Dexsuite Kuka+Allegro - Dexterous Reorient RL
echo  Task: Isaac-Dexsuite-Kuka-Allegro-Reorient-v0
echo ================================================
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%train_dexsuite_reorient.py" %*
endlocal
