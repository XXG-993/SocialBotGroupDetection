import os
import datetime
import logging
from ReadConfig import ReadConfig


class Singleton(type):
    _instances = {}

    def __new__(mcs, name, bases, namespaces, **kwargs):
        if "get_logger" not in namespaces:
            raise TypeError("no get_logger method")
        return super().__new__(mcs, name, bases, namespaces, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MyLogger(object, metaclass=Singleton):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')
        now = datetime.datetime.now()
        dirname = ReadConfig().read_config()['log_path']
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        file_handler = logging.FileHandler(dirname + "/log_"  + now.strftime("%Y-%m-%d-%H:%M")+".log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger


# if __name__ == '__main__':
#     logger = MyLogger.__call__().get_logger()
#     logger.info("info")
