@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%GPU_ID%"=="" set GPU_ID=0
if "%EPOCHS%"=="" set EPOCHS=15
if "%P_AUG%"=="" set P_AUG=0.5
if "%COMBO%"=="" set COMBO=color_texture

if "%COMBO%"=="color_texture" (
  set TRAIN_AUG=resize_bilateral_grayscale
  set LOG_NAME=robust_color_texture
) else if "%COMBO%"=="shape_texture" (
  set TRAIN_AUG=resize_patchshuffle_bilateral
  set LOG_NAME=robust_shape_texture
) else if "%COMBO%"=="shape_color" (
  set TRAIN_AUG=resize_patchshuffle_grayscale
  set LOG_NAME=robust_shape_color
) else (
  echo Unknown COMBO=%COMBO%. Use color_texture, shape_texture, or shape_color.
  exit /b 1
)

set LOGS_DIR=%ROOT_DIR%\logs_radio_sensing\%LOG_NAME%
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

echo [train] deepglobe %COMBO% aug (p=%P_AUG%, epochs=%EPOCHS%) -> %LOGS_DIR%
python "%ROOT_DIR%\training.py" params.dataset=deepglobe params.slurm_bypass=True params.cuda_no=%GPU_ID% params.max_epochs=%EPOCHS% model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%" logging.save_checkpoint=True dataaug.train_augmentations=%TRAIN_AUG% dataaug.test_augmentations=resize dataaug.p=%P_AUG% dataaug.grid_size=4 dataaug.gray_alpha=1.0 dataaug.bilateral_d=5 dataaug.sigma_color=75 dataaug.sigma_space=75
