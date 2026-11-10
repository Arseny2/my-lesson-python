@echo off
rem Запуск торгового бота на Windows: двойной клик по этому файлу.
rem Первый запуск сам установит всё нужное (1-3 минуты).
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

if not exist ".venv\Scripts\python.exe" (
    echo Первый запуск: создаю окружение Python...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python не найден. Установите Python 3.10 или новее с https://www.python.org/downloads/
        echo При установке поставьте галочку "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo Проверяю библиотеки...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Не удалось установить библиотеки. Проверьте интернет и запустите снова.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
pause
