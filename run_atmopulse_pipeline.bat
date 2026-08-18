@echo off
echo ==========================================
echo  AtmoPulse Operative Pipeline
echo ==========================================

:: 1. Anaconda Umgebung aktivieren
call "C:\Users\liina\miniconda3\condabin\activate.bat" cee_env

cd /d "C:\Users\liina\Andreas ERA5"

:: 2. Live-Vorhersage AIFS (Dauert ca. 2 Minuten)
echo.
echo ---> Lade AIFS (Machine Learning) Vorhersage...
python aifs_ingestion.py

:: 3. Live-Vorhersage IFS (Dauert ca. 1.5 Minuten)
echo.
echo ---> Lade IFS (Physikalisches Modell) Vorhersage...
python ifs_ingestion.py

:: 4. ERA5 Historie (Dauert ca. 22 Minuten)
echo.
echo ---> Aktualisiere ERA5 Baseline...
:: Das 'cmd /c' fängt einen eventuellen Absturz/Exit des ERA5-Skripts ab!
cmd /c run_era5_update.bat

echo.
echo ==========================================
echo Pipeline vollstaendig durchgelaufen!
echo ==========================================
pause