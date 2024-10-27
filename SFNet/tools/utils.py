import sys
from tabulate import tabulate
# from mmcv.runner import get_dist_info


class __redirection__:
    def __init__(self, mode='console', file_path=None):
        assert mode in ['console', 'file', 'both']

        self.mode = mode
        self.buff = ''
        self.__console__ = sys.stdout

        self.file = None
        if file_path is not None and mode != 'console':
            try:
                self.file = open(file_path, "w", buffering=1)
            except OSError:
                print('Fail to open log_file: {}'.format(
                    file_path))

    def write(self, output_stream):
        self.buff += output_stream
        if self.mode == 'console':
            self.to_console(output_stream)
        elif self.mode == 'file':
            self.to_file(output_stream)
        elif self.mode == 'both':
            self.to_console(output_stream)
            self.to_file(output_stream)

    def to_console(self, content):
        sys.stdout = self.__console__
        print(content, end='')
        sys.stdout = self

    def to_file(self, content):
        if self.file is not None:
            sys.stdout = self.file
            print(content, end='')
            sys.stdout = self

    def all_to_console(self, flush=False):
        sys.stdout = self.__console__
        print(self.buff, end='')
        sys.stdout = self

    def all_to_file(self, file_path=None, flush=True):
        if file_path is not None:
            self.open(file_path)
        if self.file is not None:
            sys.stdout = self.file
            print(self.buff, end='')
            sys.stdout = self
            # self.file.close()

    def open(self, file_path):
        try:
            self.file = open(file_path, "w", buffering=1)
        except OSError:
            print('Fail to open log_file: {}'.format(
                file_path))

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def flush(self):
        self.buff = ''

    def reset(self):
        sys.stdout = self.__console__


def mertics2markdown(metrics, exp_id):
    # 简化键名
    new_keys = {'accuracy/top1': 'OA'}
    # 移除 'single-label/' 并更新其余键
    formatted_metrics = {}
    for key, value in metrics.items():
        new_key = new_keys.get(key, key.replace('single-label/', ''))
        formatted_metrics[new_key] = f"{value:.2f}"  # 格式化数值保留两位小数

    # Save metrics as markdown
    headers = ['Experiment ID'] + list(formatted_metrics.keys())
    content = [(exp_id, ) + tuple(formatted_metrics.values())]
    markdown_table = tabulate(content, headers=headers, tablefmt="pipe")

    return markdown_table
