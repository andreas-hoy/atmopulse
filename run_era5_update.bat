@echo off
:: Change to the project directory
cd /d "C:\Users\liina\Andreas ERA5"

:: Parent pipeline already activated cee_env. Re-activating in a nested
:: cmd prints "Das System kann den angegebenen Pfad nicht finden" from
:: conda hooks and is unnecessary. Activate only when run standalone.
if /I not "%CONDA_DEFAULT_ENV%"=="cee_env" (
    call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env
)

python era5_daily_updater.py
