# Запуск программы: python main.py [путь_к_config.yaml].
# Настраивает журнал и запускает приложение из src/app.py.

import sys

from src.app import App
from src.reporting import setup_logging


def main() -> None:
    # Настраивает логирование и запускает приложение.
    setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    App(config_path).run()


if __name__ == "__main__":
    main()
