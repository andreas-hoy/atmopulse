@echo off
echo ==========================================
echo  AtmoPulse operational pipeline
echo ==========================================

:: 1. Activate the Conda environment
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env

cd /d "C:\Users\liina\Andreas ERA5"

:: 2. Live AIFS forecast (~2 minutes)
echo.
echo ---> Downloading AIFS (machine-learning) forecast...
python aifs_ingestion.py

:: 3. Live IFS forecast (~1.5 minutes)
echo.
echo ---> Downloading IFS (physics) forecast...
python ifs_ingestion.py

:: 4. ERA5 history (~22 minutes)
echo.
echo ---> Updating ERA5 baseline...
:: cmd /c isolates an ERA5-script crash/exit from this parent batch
cmd /c run_era5_update.bat

echo.
echo ==========================================
echo Pipeline finished.
echo ==========================================
pause
