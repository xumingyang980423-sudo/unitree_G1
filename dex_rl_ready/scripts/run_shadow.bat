@echo off
REM Train Shadow Hand in-hand cube reorientation
setlocal
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  Shadow Hand - Cube Reorientation RL Training
echo  Task: Isaac-Repose-Cube-Shadow-Direct-v0
echo ================================================
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%train_shadow.py" %*
endlocal
