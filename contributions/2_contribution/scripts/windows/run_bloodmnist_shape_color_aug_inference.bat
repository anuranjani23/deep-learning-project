@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%GPU_ID%"=="" set GPU_ID=0
if "%EXP_DIR%"=="" set EXP_DIR=%ROOT_DIR%\logs_medical\shape_color_aug

python "%ROOT_DIR%\reliance_protocol.py" -d bloodmnist -m resnet50 -l "%EXP_DIR%" --cuda-no %GPU_ID%
