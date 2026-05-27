@echo off
REM Quick keyboard teleop test - Franka robot (no recording)
setlocal
set "ISAACLAB_DIR=E:\Issac_sim\IsaacLab"

echo ================================================
echo  Quick Teleop Test - Franka Robot
echo  Task: Isaac-Lift-Cube-Franka-IK-Rel-v0
echo ================================================
echo Controls: W/S A/D Q/E move, Z/X T/G C/V rotate, K gripper, R reset
echo.

cd /d "%ISAACLAB_DIR%"
set OMNI_KIT_ACCEPT_EULA=YES

call .\isaaclab.bat -p scripts/environments/teleoperation/teleop_se3_agent.py ^
    --task Isaac-Lift-Cube-Franka-IK-Rel-v0 ^
    --num_envs 1 ^
    --teleop_device keyboard

endlocal
