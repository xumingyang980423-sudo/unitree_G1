@echo off
REM Replay recorded demonstrations for verification

setlocal
set "PROJECT_ROOT=%~dp0.."
set "DATASET_DIR=%PROJECT_ROOT%\datasets"
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"

echo ================================================
echo  Replay Demonstrations - Franka Pick and Place
echo ================================================

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p scripts/tools/replay_demos.py ^
    --task Isaac-Lift-Cube-Franka-IK-Rel-v0 ^
    --device cpu ^
    --dataset_file "%DATASET_DIR%\franka_demos.hdf5"

endlocal
