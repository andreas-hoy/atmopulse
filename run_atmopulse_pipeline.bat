@echo off
setlocal EnableExtensions
echo ==========================================
echo  AtmoPulse operational pipeline
echo ==========================================

:: 1. Activate the Conda environment FIRST. activate.bat does its own
:: setlocal/endlocal; delayed expansion is intentionally not used below.
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env
cd /d "C:\Users\liina\Andreas ERA5"

:: 2. Timestamped log in ERA5_ClimateTool\Pipeline_Logs
set "LOGDIR=%CD%\ERA5_ClimateTool\Pipeline_Logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "STAMP="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "STAMP=%%I"
if not defined STAMP set "STAMP=%RANDOM%"
set "ATMOPULSE_LOG_FILE=%LOGDIR%\pipeline_%STAMP%.log"

echo ===== AtmoPulse pipeline %STAMP% =====> "%ATMOPULSE_LOG_FILE%"
echo.
echo Log file: %ATMOPULSE_LOG_FILE%
echo.

set "total_start_time=%time%"

:: 3. Live AIFS — do not wrap in cmd /c (inherits ATMOPULSE_LOG_FILE; avoids extra cmd path errors)
echo.
echo ---> Downloading AIFS (machine-learning) forecast...
call :tee ---> Downloading AIFS (machine-learning) forecast...
set "aifs_start=%time%"
python aifs_ingestion.py
set "AIFS_RC=%ERRORLEVEL%"
if not "%AIFS_RC%"=="0" (
    echo [WARNING] AIFS pipeline failed or timed out. Defaulting to IFS physics model.
    echo           Subsequent ERA5, Zarr, and precompute steps will still run.
    call :tee [WARNING] AIFS failed, RC=%AIFS_RC%. Defaulting to IFS.
) else (
    echo [SUCCESS] AIFS ingestion completed.
    call :tee [SUCCESS] AIFS ingestion completed.
)
set "aifs_end=%time%"

:: 4. Live IFS — always attempted
echo.
echo ---> Downloading IFS (physics) forecast...
call :tee ---> Downloading IFS (physics) forecast...
set "ifs_start=%time%"
python ifs_ingestion.py
set "IFS_RC=%ERRORLEVEL%"
if not "%IFS_RC%"=="0" (
    echo [ERROR] IFS pipeline failed. Check logs.
    echo         Continuing with ERA5 and Zarr steps.
    call :tee [ERROR] IFS failed, RC=%IFS_RC%.
) else (
    echo [SUCCESS] IFS ingestion completed.
    call :tee [SUCCESS] IFS ingestion completed.
)
set "ifs_end=%time%"

:: 5. ERA5 history
echo.
echo ---> Updating ERA5 baseline...
call :tee ---> Updating ERA5 baseline...
set "era5_start=%time%"
call run_era5_update.bat
set "era5_end=%time%"

:: 6. Zarr archive append
echo.
echo ---> Appending new ERA5 data to Zarr archive...
call :tee ---> Appending new ERA5 data to Zarr archive...
set "zarr_start=%time%"
python batch_update_zarr.py
set "ZARR_RC=%ERRORLEVEL%"
if not "%ZARR_RC%"=="0" (
    echo [ERROR] Zarr append failed. Check logs.
    call :tee [ERROR] Zarr failed, RC=%ZARR_RC%.
) else (
    echo [SUCCESS] Zarr archive updated.
    call :tee [SUCCESS] Zarr archive updated.
)
set "zarr_end=%time%"

:: 7. Pre-computation
echo.
echo ---> Precomputing spatial footprints and Top-10 rankings...
call :tee ---> Precomputing spatial footprints and Top-10 rankings...
set "precomp_start=%time%"
python batch_precompute_analytics.py
set "PRECOMP_RC=%ERRORLEVEL%"
if not "%PRECOMP_RC%"=="0" (
    echo [ERROR] Pre-computation failed. Check logs.
    call :tee [ERROR] Precompute failed, RC=%PRECOMP_RC%.
) else (
    echo [SUCCESS] Parquet binaries generated.
    call :tee [SUCCESS] Parquet binaries generated.
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
call :tee Pipeline finished. Log: %ATMOPULSE_LOG_FILE%
pause
exit /b 0

:tee
if defined ATMOPULSE_LOG_FILE (
    echo %*>> "%ATMOPULSE_LOG_FILE%"
)
goto :eof
