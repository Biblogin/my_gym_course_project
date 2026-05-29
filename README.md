Для запуска проекта на Unix-подобной системе или WSL (Windows SubSystem for Linux) выполните следующие команды:

Для установки необходимых библиотек выполните следующее:
    Перейдите в директорию проекта
    (для установки на Debian/Ubuntu системах, на других используйте свой пакетный менеджер)

    sudo apt install python3.12-venv -y
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

После чего запустить main.py:

    python main.py