@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul || (echo Не найден Python. Установите Python 3.11 с https://www.python.org/downloads/ и отметьте "Add python.exe to PATH" & pause & exit /b 1)
if not exist .venv (
  echo Первый запуск: создаю окружение и ставлю библиотеки, 5-10 минут...
  python -m venv .venv || (pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements-app.txt || (echo Ошибка установки библиотек & pause & exit /b 1)
)
if not exist weights\best.pt (echo Нет файла weights\best.pt - положите туда веса модели & pause & exit /b 1)
set EX=
if exist examples set EX=--examples-dir examples
echo Запускаю интерфейс: http://127.0.0.1:7860  (остановить - закрыть это окно)
.venv\Scripts\python app.py --ckpt weights\best.pt %EX% --inbrowser
pause
