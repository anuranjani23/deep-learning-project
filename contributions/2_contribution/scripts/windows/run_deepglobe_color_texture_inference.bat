@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%GPU_ID%"=="" set GPU_ID=0
if "%EXP_DIR%"=="" set EXP_DIR=%ROOT_DIR%\logs_radio_sensing\robust_color_texture

python "%ROOT_DIR%\reliance_protocol.py" -d deepglobe -m resnet50 -l "%EXP_DIR%" --cuda-no %GPU_ID%
