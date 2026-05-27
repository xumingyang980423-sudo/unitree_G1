@echo off
REM Install Isaac Lab official CUDA PyTorch (2.7.0+cu128)
setlocal
set "PY=E:\Issac_sim\IsaacLab\_isaac_sim\kit\python\python.exe"
echo ================================================
echo  Install PyTorch 2.7.0+cu128 for Isaac Lab
echo  Download size ~3.3 GB, may take several minutes
echo ================================================
"%PY%" -m pip uninstall -y torch torchvision torchaudio
"%PY%" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
echo.
echo Verifying CUDA...
"%PY%" -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
endlocal
pause
