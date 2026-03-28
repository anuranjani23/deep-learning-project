@echo off
setlocal

set ROOT_DIR=%~dp0..\..
for %%A in ("%ROOT_DIR%") do set ROOT_DIR=%%~fA

if "%CAPTIONS%"=="" set CAPTIONS=%ROOT_DIR%\dataset\coco\coco_train2017_subset.jsonl
if "%MAX_SAMPLES%"=="" set MAX_SAMPLES=200
if "%PROMPT_SETS%"=="" set PROMPT_SETS=clip_basic,shape,texture,color
if "%LOG_DIR%"=="" set LOG_DIR=%ROOT_DIR%\logs_prompt_eng

if not exist "%CAPTIONS%" (
  echo Captions file not found: %CAPTIONS%
  exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :run_mode caption
call :run_mode noun
call :run_mode short

echo All prompt-engineering comparisons saved under %LOG_DIR%
exit /b 0

:run_mode
set LABEL_MODE=%1
set OUT_CSV=%LOG_DIR%\coco_prompt_%LABEL_MODE%.csv
set OUT_LOG=%LOG_DIR%\coco_prompt_%LABEL_MODE%.txt

echo [coco prompt] label_mode=%LABEL_MODE% max_samples=%MAX_SAMPLES% prompt_sets=%PROMPT_SETS%
if "%DEVICE%"=="" (
  powershell -NoProfile -Command "python '%ROOT_DIR%\\scripts\\test_coco_prompt_engineering.py' --captions '%CAPTIONS%' --max-samples '%MAX_SAMPLES%' --prompt-sets '%PROMPT_SETS%' --label-mode '%LABEL_MODE%' --save-csv '%OUT_CSV%' 2>&1 | Tee-Object -FilePath '%OUT_LOG%'"
) else (
  powershell -NoProfile -Command "python '%ROOT_DIR%\\scripts\\test_coco_prompt_engineering.py' --captions '%CAPTIONS%' --max-samples '%MAX_SAMPLES%' --prompt-sets '%PROMPT_SETS%' --label-mode '%LABEL_MODE%' --device '%DEVICE%' --save-csv '%OUT_CSV%' 2>&1 | Tee-Object -FilePath '%OUT_LOG%'"
)
exit /b 0
