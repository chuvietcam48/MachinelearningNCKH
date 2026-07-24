@echo off
echo ==========================================
echo Fix Python Environment and Install Package
echo ==========================================
echo.
echo [1/2] Downloading missing standard library file (stringprep.py)...
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/python/cpython/3.12/Lib/stringprep.py' -OutFile '.\.venv\Lib\site-packages\stringprep.py'"

echo.
echo [2/2] Installing google-genai...
.\.venv\Scripts\python.exe -m pip install -U google-genai

echo.
echo Done! Please run your preflight and smoke test commands now.
pause
