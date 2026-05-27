@echo off
setlocal
cd /d "%~dp0"

REM ---- Isaac Sim paths ----
set "ISAAC_ROOT=E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64"
set "ISAAC_KIT=%ISAAC_ROOT%\kit"

REM ---- Check Isaac Sim ----
if not exist "%ISAAC_KIT%" (
    echo [ERROR] Isaac Sim kit not found: %ISAAC_KIT%
    pause
    exit /b 1
)

REM ---- Required environment variables ----
set "CARB_APP_PATH=%ISAAC_KIT%"
set "ISAAC_PATH=%ISAAC_ROOT%"
set "EXP_PATH=%ISAAC_KIT%\apps"

REM ---- PYTHONPATH (site triggers sitecustomize.py) ----
set "PYTHONPATH=%ISAAC_ROOT%\site;%PYTHONPATH%"

REM ---- Python executable ----
set "PYTHON_EXE=%ISAAC_KIT%\python\python.exe"

echo ================================================================
echo   Unitree G1 - Isaac Lab Locomotion
echo ================================================================
echo   Isaac Sim:  %ISAAC_ROOT%
echo ================================================================
echo.
echo Choose an action:
echo   [1] Train - Flat terrain (recommended)
echo   [2] Train - Rough terrain
echo   [3] Play with trained policy
echo   [4] Quick test with random actions
echo.
set /p CHOICE="Enter choice [1-4]: "

if "%CHOICE%"=="1" call "%PYTHON_EXE%" "%~dp0scripts\train.py" --task Isaac-Velocity-Flat-G1-v0 & goto :done
if "%CHOICE%"=="2" call "%PYTHON_EXE%" "%~dp0scripts\train.py" --task Isaac-Velocity-Rough-G1-v0 & goto :done
if "%CHOICE%"=="3" call "%PYTHON_EXE%" "%~dp0scripts\play.py" --task Isaac-Velocity-Flat-G1-Play-v0 & goto :done
if "%CHOICE%"=="4" call "%PYTHON_EXE%" "%~dp0scripts\random_play.py" --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 4 & goto :done
echo [ERROR] Invalid choice.

:done
endlocal
pause
exit /b 0
