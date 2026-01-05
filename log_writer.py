import logging
import os
import sys
from logging import handlers


class MyLogger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, file_path="", filename="", level='info', when='D', back_count=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        if file_path != "":
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        filename = os.path.join(file_path, filename)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        format_str = logging.Formatter(fmt)  # 设置日志格式
        stream_handler = logging.StreamHandler()  # 往屏幕上输出
        stream_handler.setFormatter(format_str)  # 设置屏幕上显示的格式
        # 指定间隔时间自动生成文件的处理器
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=back_count,
                                               encoding='utf-8')
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，
        # backupCount是备份文件的个数，如果超过这个个数，就会自动删除，
        # when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(stream_handler)  # 把对象加到logger里
        self.logger.addHandler(th)


def example():
    logw = MyLogger(filename="./results.txt")
    logw.logger.info("write to file")


if __name__ == '__main__':
    example()
