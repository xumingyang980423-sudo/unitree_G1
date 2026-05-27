@echo off
REM G1 keyboard teleop test - right arm + inspire hand grasping
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"

echo ================================================
echo  G1 Inspire Hand - Keyboard Teleop Test
echo  Task: Isaac-PickPlace-G1-InspireFTP-Abs-v0
echo ================================================
echo Controls:
echo   W/S: forward/back    A/D: left/right    Q/E: up/down
echo   Z/X: rotate X        T/G: rotate Y      C/V: rotate Z
echo   K  : toggle right hand grip    R: reset
echo ================================================
echo.

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p "%SCRIPT_DIR%test_teleop_g1.py" --enable_pinocchio %*
endlocal
