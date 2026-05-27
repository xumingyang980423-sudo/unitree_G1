@echo off
REM View G1 Inspire Hand + Red Block scene (Unitree official config)
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"
echo ================================================
echo  G1 + Inspire Hand + Table + Red Block
echo  Unitree Official Scene
echo ================================================
cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES
call .\isaaclab.bat -p "%SCRIPT_DIR%view_scene.py" --device cuda:0 %*
endlocal
