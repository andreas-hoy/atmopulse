@echo off
setlocal EnableExtensions
echo ==========================================
echo  AtmoPulse operational pipeline
echo ==========================================

:: 1. Activate the Conda environment
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env
if errorlevel 1 (
    echo [ERROR] Conda activate failed. Is cee_env installed?
    goto :hold
)
cd /d "C:\Users\liina\Andreas ERA5"
if errorlevel 1 (
    echo [ERROR] Could not cd to the project folder.
    goto :hold
)

:: 2. Timestamped log. STAMP is set on its own line so %STAMP% expands correctly.
set "LOGDIR=%CD%\ERA5_ClimateTool\Pipeline_Logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "STAMP="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "STAMP=%%I"
if not defined STAMP set "STAMP=%RANDOM%"
set "ATMOPULSE_LOG_FILE=%LOGDIR%\pipeline_%STAMP%.log"

>>"%ATMOPULSE_LOG_FILE%" echo ===== AtmoPulse pipeline %STAMP% =====
echo.
echo Log file: %ATMOPULSE_LOG_FILE%
echo.

set "total_start_time=%time%"

:: 3. Live AIFS forecast
echo.
echo --- Downloading AIFS machine-learning forecast...
>>"%ATMOPULSE_LOG_FILE%" echo --- Downloading AIFS machine-learning forecast...
set "aifs_start=%time%"
python aifs_ingestion.py
set "AIFS_RC=%ERRORLEVEL%"
if not "%AIFS_RC%"=="0" (
    echo [WARNING] AIFS pipeline failed or timed out. Defaulting to IFS physics model.
    echo           Subsequent ERA5, Zarr, and precompute steps will still run.
    >>"%ATMOPULSE_LOG_FILE%" echo [WARNING] AIFS failed, RC=%AIFS_RC%. Defaulting to IFS.
) else (
    echo [SUCCESS] AIFS ingestion completed.
    >>"%ATMOPULSE_LOG_FILE%" echo [SUCCESS] AIFS ingestion completed.
)
set "aifs_end=%time%"

:: 4. Live IFS forecast — always attempted
echo.
echo --- Downloading IFS physics forecast...
>>"%ATMOPULSE_LOG_FILE%" echo --- Downloading IFS physics forecast...
set "ifs_start=%time%"
python ifs_ingestion.py
set "IFS_RC=%ERRORLEVEL%"
if not "%IFS_RC%"=="0" (
    echo [ERROR] IFS pipeline failed. Check logs.
    echo         Continuing with ERA5 and Zarr steps.
    >>"%ATMOPULSE_LOG_FILE%" echo [ERROR] IFS failed, RC=%IFS_RC%.
) else (
    echo [SUCCESS] IFS ingestion completed.
    >>"%ATMOPULSE_LOG_FILE%" echo [SUCCESS] IFS ingestion completed.
)
set "ifs_end=%time%"

:: 5. ERA5 history
echo.
echo --- Updating ERA5 baseline...
>>"%ATMOPULSE_LOG_FILE%" echo --- Updating ERA5 baseline...
set "era5_start=%time%"
call run_era5_update.bat
set "era5_end=%time%"

:: 6. Zarr archive append
echo.
echo --- Appending new ERA5 data to Zarr archive...
>>"%ATMOPULSE_LOG_FILE%" echo --- Appending new ERA5 data to Zarr archive...
set "zarr_start=%time%"
python batch_update_zarr.py
set "ZARR_RC=%ERRORLEVEL%"
if not "%ZARR_RC%"=="0" (
    echo [ERROR] Zarr append failed. Check logs.
    >>"%ATMOPULSE_LOG_FILE%" echo [ERROR] Zarr failed, RC=%ZARR_RC%.
) else (
    echo [SUCCESS] Zarr archive updated.
    >>"%ATMOPULSE_LOG_FILE%" echo [SUCCESS] Zarr archive updated.
)
set "zarr_end=%time%"

:: 7. Pre-computation
echo.
echo --- Precomputing spatial footprints and Top-10 rankings...
>>"%ATMOPULSE_LOG_FILE%" echo --- Precomputing spatial footprints and Top-10 rankings...
set "precomp_start=%time%"
python batch_precompute_analytics.py
set "PRECOMP_RC=%ERRORLEVEL%"
if not "%PRECOMP_RC%"=="0" (
    echo [ERROR] Pre-computation failed. Check logs.
    >>"%ATMOPULSE_LOG_FILE%" echo [ERROR] Precompute failed, RC=%PRECOMP_RC%.
) else (
    echo [SUCCESS] Parquet binaries generated.
    >>"%ATMOPULSE_LOG_FILE%" echo [SUCCESS] Parquet binaries generated.
)
set "precomp_end=%time%"

echo.
echo ==========================================
echo Pipeline finished.
echo Execution Timestamps:
echo AIFS    Start : %aifs_start% ^| End: %aifs_end% ^| RC: %AIFS_RC%
echo IFS     Start : %ifs_start%  ^| End: %ifs_end%  ^| RC: %IFS_RC%
echo ERA5    Start : %era5_start% ^| End: %era5_end%
echo ZARR    Start : %zarr_start% ^| End: %zarr_end% ^| RC: %ZARR_RC%
echo PRECOMP Start : %precomp_start% ^| End: %precomp_end% ^| RC: %PRECOMP_RC%
echo Log file: %ATMOPULSE_LOG_FILE%
echo ==========================================
>>"%ATMOPULSE_LOG_FILE%" echo Pipeline finished. Log: %ATMOPULSE_LOG_FILE%

:hold
echo.
pause
