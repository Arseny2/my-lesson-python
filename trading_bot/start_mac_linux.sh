#!/usr/bin/env bash
# Запуск торгового бота на Mac / Linux:  ./start_mac_linux.sh   (или: bash start_mac_linux.sh)
# Первый запуск сам установит всё нужное (1-3 минуты).
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python не найден. Установите Python 3.10 или новее с https://www.python.org/downloads/"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение Python..."
  python3 -m venv .venv
fi
echo "Проверяю библиотеки..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
exec .venv/bin/python main.py "$@"
