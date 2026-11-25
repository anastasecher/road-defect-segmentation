#!/bin/bash
cd "$(dirname "$0")"
PY=python3
command -v $PY >/dev/null 2>&1 || { echo "Не найден python3. Установите Python 3.11 с https://www.python.org/downloads/"; read -p "Enter для выхода"; exit 1; }
if [ ! -d .venv ]; then
  echo "Первый запуск: создаю окружение и ставлю библиотеки (5–10 минут)..."
  $PY -m venv .venv || exit 1
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-app.txt || { echo "Ошибка установки библиотек"; read -p "Enter"; exit 1; }
fi
[ -f weights/best.pt ] || { echo "Нет файла weights/best.pt — положите туда веса модели"; read -p "Enter"; exit 1; }
EX=""; [ -d examples ] && EX="--examples-dir examples"
echo "Запускаю интерфейс: http://127.0.0.1:7860 (остановить — Ctrl+C или закрыть окно)"
.venv/bin/python app.py --ckpt weights/best.pt $EX --inbrowser
