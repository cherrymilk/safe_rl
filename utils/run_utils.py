import time
import os.path as osp
from datetime import datetime


DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))), 'data')


# swb:这个函数的功能就是说，需要设置logger所需要的参数啊！！
def setup_logger_path(exp_name, data_dir, datestamp: [bool, str], **kwargs):
    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    # swb:建立一个二级目录啊！！
    assert datestamp is (True or False) or type(datestamp) == str
    if datestamp is True:
        hms_time = time.strftime("%H-%M-%S")
        # hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    elif type(datestamp) == str:
        hms_time = datestamp
    elif datestamp is False:
        hms_time = ""
    else:
        raise NotImplementedError
    # sub_folder = ''.join([hms_time, '-', exp_name])
    sub_folder = hms_time
    for key, item in kwargs.items():
        sub_folder += f"-{key}-{item}"
    relpath = osp.join(relpath, sub_folder)
    output_dir = osp.join(data_dir, relpath)
    return output_dir


# swb:这个函数的功能就是说，需要设置logger所需要的参数啊！！
def setup_logger_path_swb(datestamp: [bool, str], data_dir: str, **kwargs):
    # swb:确保是有时间戳的啊！！
    assert datestamp is True or type(datestamp) == str
    if datestamp is True:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        now = datestamp
    url = osp.join(data_dir, now)
    for key, item in kwargs.items():
        url += "-{}-{}".format(key, item)
    return url


if __name__ == "__main__":
    print(setup_logger_path_swb(datestamp=True, data_dir="../data/ppo/", parallel_num=5))


