@echo off
:: Change to the project directory
cd /d "C:\Users\liina\Andreas ERA5"

:: 1. Activate Miniconda environment cee_env
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env

:: 2. Run the ERA5 daily updater
python era5_daily_updater.py
