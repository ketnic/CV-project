import sys


def inline_print(text):
    """
        Печатает в текущую строку
    """
    sys.stdout.write("\r" + text)
    sys.stdout.flush()


def complete_inline_print():
    """
        Функция, которую необходимо вызвать после inline_print
    """
    print()
