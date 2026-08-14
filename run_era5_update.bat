@echo off
:: In das Projektverzeichnis wechseln
cd /d "C:\Users\liina\Andreas ERA5"

:: 1. Miniconda-Aktivierungsskript aufrufen und cee_env übergeben
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env

:: 2. Updater-Skript ausführen
python era5_daily_updater.py