@echo off
echo ==========================================
echo  AtmoPulse operational pipeline
echo ==========================================

:: 1. Activate the Conda environment
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env

cd /d "C:\Users\liina\Andreas ERA5"

:: Initialize execution timers
set "total_start_time=%time%"

:: 2. Live AIFS forecast (~2 minutes)
echo.
echo ---> Downloading AIFS (machine-learning) forecast...
set "aifs_start=%time%"
python aifs_ingestion.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] AIFS pipeline failed or timed out. Proceeding to IFS ingestion...
) else (
    echo [SUCCESS] AIFS ingestion completed.
)
set "aifs_end=%time%"

:: 3. Live IFS forecast (~1.5 minutes)
echo.
echo ---> Downloading IFS (physics) forecast...
set "ifs_start=%time%"
python ifs_ingestion.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] IFS pipeline failed. Check logs.
) else (
    echo [SUCCESS] IFS ingestion completed.
)
set "ifs_end=%time%"

:: 4. ERA5 history (~22 minutes)
echo.
echo ---> Updating ERA5 baseline...
set "era5_start=%time%"
:: cmd /c isolates an ERA5-script crash/exit from this parent batch
cmd /c run_era5_update.bat
set "era5_end=%time%"

echo.
echo ==========================================
echo Pipeline finished.
echo Execution Timestamps:
echo AIFS Start : %aifs_start% ^| End: %aifs_end%
echo IFS  Start : %ifs_start%  ^| End: %ifs_end%
echo ERA5 Start : %era5_start% ^| End: %era5_end%
echo ==========================================
pause
