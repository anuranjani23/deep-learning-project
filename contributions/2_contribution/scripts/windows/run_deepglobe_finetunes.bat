@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%LOGS_DIR%"=="" set LOGS_DIR=%ROOT_DIR%\logs_radio_sensing
if "%GPU_IDS%"=="" set GPU_IDS=0
if "%DOWNLOAD%"=="" set DOWNLOAD=1
if "%DEEPGLOBE_DST%"=="" set DEEPGLOBE_DST=%ROOT_DIR%\dataset\tomburgert

for /f "tokens=1-3 delims=," %%A in ("%GPU_IDS%") do (
  set GPU0=%%A
  set GPU1=%%B
  set GPU2=%%C
)
if "%GPU1%"=="" set GPU1=%GPU0%
if "%GPU2%"=="" set GPU2=%GPU0%

if "%DOWNLOAD%"=="1" (
  if not "%DEEPGLOBE_SRC%"=="" (
    echo [prep] DeepGlobe from %DEEPGLOBE_SRC% -> %DEEPGLOBE_DST%
    python "%ROOT_DIR%\scripts\preprocess_deepglobe.py" --source_path "%DEEPGLOBE_SRC%" --destination_path "%DEEPGLOBE_DST%"
  ) else (
    echo [prep] DeepGlobe via kagglehub -> %DEEPGLOBE_DST%
    python "%ROOT_DIR%\scripts\preprocess_deepglobe.py" --use_kagglehub --destination_path "%DEEPGLOBE_DST%"
  )
)

if not exist "%LOGS_DIR%\baseline" mkdir "%LOGS_DIR%\baseline"
if not exist "%LOGS_DIR%\shape_removed" mkdir "%LOGS_DIR%\shape_removed"
if not exist "%LOGS_DIR%\texture_removed" mkdir "%LOGS_DIR%\texture_removed"
if not exist "%LOGS_DIR%\color_removed" mkdir "%LOGS_DIR%\color_removed"

echo [train] baseline
python "%ROOT_DIR%\training.py" params.dataset=deepglobe params.slurm_bypass=True params.cuda_no=%GPU0% params.max_epochs=5 model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%\baseline" logging.save_checkpoint=True dataaug.train_augmentations=resize dataaug.test_augmentations=resize

echo [train] shape_removed
python "%ROOT_DIR%\training.py" params.dataset=deepglobe params.slurm_bypass=True params.cuda_no=%GPU1% params.max_epochs=5 model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%\shape_removed" logging.save_checkpoint=True dataaug.train_augmentations=resize_patchshuffle dataaug.test_augmentations=resize_patchshuffle dataaug.grid_size=4

echo [train] texture_removed
python "%ROOT_DIR%\training.py" params.dataset=deepglobe params.slurm_bypass=True params.cuda_no=%GPU2% params.max_epochs=5 model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%\texture_removed" logging.save_checkpoint=True dataaug.train_augmentations=resize_bilateral dataaug.test_augmentations=resize_bilateral dataaug.bilateral_d=5 dataaug.sigma_color=75 dataaug.sigma_space=75

echo [train] color_removed
python "%ROOT_DIR%\training.py" params.dataset=deepglobe params.slurm_bypass=True params.cuda_no=%GPU0% params.max_epochs=5 model.name=resnet50 model.timm_pretrained=True logging.exp_dir="%LOGS_DIR%\color_removed" logging.save_checkpoint=True dataaug.train_augmentations=resize_grayscale dataaug.test_augmentations=resize_grayscale dataaug.gray_alpha=1.0

echo All training jobs finished.
