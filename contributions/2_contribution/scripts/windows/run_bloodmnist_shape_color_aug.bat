@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%LOGS_DIR%"=="" set LOGS_DIR=%ROOT_DIR%\logs_medical\shape_color_aug
if "%GPU_ID%"=="" set GPU_ID=0
if "%P_AUG%"=="" set P_AUG=0.5
if "%EPOCHS%"=="" set EPOCHS=10

if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

echo [train] bloodmnist shape+color aug (p=%P_AUG%) -> %LOGS_DIR%
python "%ROOT_DIR%\training.py" params.dataset=bloodmnist params.slurm_bypass=True params.cuda_no=%GPU_ID% params.max_epochs=%EPOCHS% model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%" logging.save_checkpoint=True dataaug.train_augmentations=resize_patchshuffle_grayscale dataaug.test_augmentations=resize dataaug.grid_size=4 dataaug.gray_alpha=1.0 dataaug.p=%P_AUG%
