@echo off
REM Record Franka pick-and-place demonstrations with keyboard teleop
REM Saves dataset to: g1_grasp_rl\datasets\franka_demos.hdf5

setlocal
set "PROJECT_ROOT=%~dp0.."
set "DATASET_DIR=%PROJECT_ROOT%\datasets"
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"

if not exist "%DATASET_DIR%" mkdir "%DATASET_DIR%"

echo ================================================
echo  Record Demonstrations - Franka Pick and Place
echo  Task: Isaac-Lift-Cube-Franka-IK-Rel-v0
echo  Save to: %DATASET_DIR%\franka_demos.hdf5
echo ================================================
echo.
echo Controls:
echo   W/S: forward/back    A/D: left/right    Q/E: up/down
echo   Z/X: rotate X        T/G: rotate Y      C/V: rotate Z
echo   K  : toggle gripper  R  : reset/retry
echo   Move the cube to target (green sphere), then release
echo   You need 10 successful demos for training.
echo ================================================
echo.

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p scripts/tools/record_demos.py ^
    --task Isaac-Lift-Cube-Franka-IK-Rel-v0 ^
    --device cpu ^
    --teleop_device keyboard ^
    --dataset_file "%DATASET_DIR%\franka_demos.hdf5" ^
    --num_demos 10

echo.
echo Done! Dataset saved to: %DATASET_DIR%\franka_demos.hdf5
endlocal
