import os


def make_file(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
