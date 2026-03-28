@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%LOGS_DIR%"=="" set LOGS_DIR=%ROOT_DIR%\logs_medical
if "%GPU_ID%"=="" set GPU_ID=0

echo [protocol] baseline
python "%ROOT_DIR%\reliance_protocol.py" -d bloodmnist -m resnet50 -l "%LOGS_DIR%\baseline" --cuda-no %GPU_ID%

echo [protocol] shape_removed
python "%ROOT_DIR%\reliance_protocol.py" -d bloodmnist -m resnet50 -l "%LOGS_DIR%\shape_removed" --cuda-no %GPU_ID%

echo [protocol] texture_removed
python "%ROOT_DIR%\reliance_protocol.py" -d bloodmnist -m resnet50 -l "%LOGS_DIR%\texture_removed" --cuda-no %GPU_ID%

echo [protocol] color_removed
python "%ROOT_DIR%\reliance_protocol.py" -d bloodmnist -m resnet50 -l "%LOGS_DIR%\color_removed" --cuda-no %GPU_ID%

echo All inference runs finished.
