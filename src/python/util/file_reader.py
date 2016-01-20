from os import listdir
from os.path import isfile, join


def get_files(path, end=""):
    file_paths = []
    for f in listdir(path):
        if isfile(join(path, f)):
            if f.endswith(end):
                file_paths.append(join(path, f))
        else:
            for digit_path in get_files(join(path, f), end):
                file_paths.append(digit_path)
    return file_paths
