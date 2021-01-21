import sys
import os
import psutil


def print_line(char='=', len=79, file=None):
    if file is None:
        file = sys.stdout
    print(char * len, file=file)


def print_memory_usage(prefix=None):
    if prefix is None:
        prefix = 'Memory used: '
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_usage = py.memory_info()[0]/2.**30
    print('{}{:.2f}GB'.format(prefix, memory_usage))
