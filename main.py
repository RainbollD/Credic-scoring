"""
main.py

Главный управляющий скрипт для запуска серии экспериментов и журналированием результатов.
"""
import subprocess


def run_command_shell_with_logging(cmd: str):
    """
    Запуск команды в shell с журналированием вывода.
    :param cmd: Команда для строки в shell.
    """
    res = ""
    process = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE)

    for line in process.stdout:
        print(line)
        res += line
    process.wait()
    return res, process.returncode


def main():
    """
    Основная функция: запускает серию экспериментов по списку конфигов,
    журналирует результаты и сохраняет их в отчеты.
    """
    configs = ["/home/lev/PycharmProjects/Credic-scoring/config/credit_scoring/all_models.yaml"]

    for config_path in configs:
        print(config_path)
        output, code = run_command_shell_with_logging(
            " ".join(["python3.12", "-m", "core.runner", "-c", config_path])
        )


if __name__ == '__main__':
    main()
