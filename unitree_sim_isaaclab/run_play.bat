@echo off
REM Play trained G1 Inspire pick-place policy
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  G1 Inspire Hand - Pick-Place Policy Play
echo ================================================
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%play.py" --device cuda:0 --smooth 1.0 %*
endlocal
