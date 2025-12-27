@echo off
if "%1"=="KEEP_OPEN" goto :START_MAIN
cmd /k ""%~f0" KEEP_OPEN"
exit /b

:START_MAIN
setlocal
echo.

:: 1. Check Python
echo [STEP 1] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    goto :FAIL
)

:: 2. Core Math - Splitting these up to identify the crasher
echo.
echo [STEP 2a] Installing Numpy...
python -m pip install "numpy>=1.20.0"
if %errorlevel% neq 0 goto :FAIL

echo.
echo [STEP 2b] Installing Scipy...
echo (This step often fails on Python 3.14 because it must compile from source)
python -m pip install "scipy>=1.7.0"
if %errorlevel% neq 0 goto :FAIL

:: 3. Visualization
echo.
echo [STEP 3] Installing Matplotlib and Astropy...
python -m pip install "matplotlib>=3.4.0" "astropy>=5.0"
if %errorlevel% neq 0 goto :FAIL

:: 4. Data Handlers
echo.
echo [STEP 4] Installing NetCDF4, and H5py
python -m pip install "netCDF4>=1.5.0" "h5py>=3.0.0"
if %errorlevel% neq 0 goto :FAIL

:: 5. Optionals
echo.
echo [STEP 5] Installing Joblib...
python -m pip install "joblib>=1.0.0"
if %errorlevel% neq 0 goto :FAIL

echo.
echo ========================================================
echo  SUCCESS: All packages installed.
echo ========================================================
echo.
echo You can now close this window.
exit /b 0

:FAIL
echo.
echo ########################################################
echo  INSTALLATION CRASHED
echo ########################################################
echo.
echo The script has stopped. The window is forced open.
echo SCROLL UP to see the red error message above.
echo.
exit /b 1