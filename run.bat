@echo off
REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Activating the conda environment
call conda activate tobaigc

REM Running the Python scripts concurrently
start python main.py --listen
start python aigcserver.py

REM Keeping the terminal open
cmd /k

