@echo off
title Clearview
cd /d "%~dp0"

echo.
echo  ========================================
echo   Clearview - Iniciando...
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo.
    echo Descarga Python desde: https://www.python.org/downloads/
    echo IMPORTANTE: Marca la casilla "Add Python to PATH" al instalar.
    echo.
    pause
    exit /b 1
)

echo [1/3] Instalando dependencias (solo la primera vez tarda un poco)...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo al instalar dependencias.
    pause
    exit /b 1
)

echo [2/3] Verificando FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [AVISO] FFmpeg no esta instalado.
    echo Sin FFmpeg no podras recortar ni guardar clips.
    echo.
    echo Instala FFmpeg de una de estas formas:
    echo   1. winget install ffmpeg
    echo   2. https://ffmpeg.org/download.html
    echo   3. O con chocolatey: choco install ffmpeg
    echo.
) else (
    echo      FFmpeg OK
)

echo [3/3] Arrancando servidor...
echo.
echo  ------------------------------------------
echo   Abre en tu navegador:
echo.
echo   http://localhost:8000
echo   http://localhost:8000/editor
echo   http://localhost:8000/create
echo  ------------------------------------------
echo.
echo  Para detener el servidor: Ctrl + C
echo.

cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
