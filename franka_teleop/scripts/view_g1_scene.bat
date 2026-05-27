@echo off
REM View G1 pick-place scene (no IK, no control, just visualization)
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"
set "SCRIPT_DIR=%~dp0"

echo ================================================
echo  G1 Pick-Place Scene Viewer
echo ================================================
echo Right-click drag to orbit camera
echo.

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p "%SCRIPT_DIR%view_g1_scene.py" %*
endlocal
