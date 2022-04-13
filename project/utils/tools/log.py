from termcolor import colored


def log(message):
    print(colored('[INFO] {}'.format(message), 'blue'))
